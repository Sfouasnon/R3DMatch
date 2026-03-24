#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef R3DMATCH_RED_SDK_ENABLED
#include "R3DSDK.h"
#include "R3DSDKMetadata.h"
using namespace R3DSDK;
#endif

namespace py = pybind11;

namespace {

bool sdk_available() {
#ifdef R3DMATCH_RED_SDK_ENABLED
    return true;
#else
    return false;
#endif
}

std::string unavailable_message() {
    return "RED SDK bridge is not available. Build the native bridge with RED_SDK_ROOT set to a local RED SDK install.";
}

std::string basename_from_path(const std::string &path) {
    const std::size_t offset = path.find_last_of("/\\");
    if (offset == std::string::npos) {
        return path;
    }
    return path.substr(offset + 1U);
}

#ifdef R3DMATCH_RED_SDK_ENABLED

std::string getenv_or_empty(const char *name) {
    const char *value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

std::string resolve_libraries_path() {
    const std::string env_lib = getenv_or_empty("RED_SDK_REDISTRIBUTABLE_DIR");
    if (!env_lib.empty()) {
        return env_lib;
    }
#ifdef R3DMATCH_RED_SDK_REDIS_DIR
    return R3DMATCH_RED_SDK_REDIS_DIR;
#else
    return ".";
#endif
}

bool red_debug_enabled() {
    const std::string value = getenv_or_empty("R3DMATCH_RED_DEBUG");
    return value == "1" || value == "true" || value == "TRUE" || value == "yes" || value == "YES";
}

void red_debug_log(const std::string &message) {
    if (red_debug_enabled()) {
        std::cerr << "[r3dmatch.red] " << message << std::endl;
    }
}

void require_supported_decode_settings(bool half_res, const std::string &colorspace, const std::string &gamma) {
    if (!half_res) {
        throw std::runtime_error("RED SDK bridge currently supports half_res=True only.");
    }
    if (colorspace != "REDWideGamutRGB" || gamma != "Log3G10") {
        throw std::runtime_error(
            "RED SDK bridge currently supports only colorspace=REDWideGamutRGB and gamma=Log3G10."
        );
    }
}

std::string bool_string(bool value) {
    return value ? "true" : "false";
}

class ScopedSdkSession {
public:
    ScopedSdkSession() : libraries_path_(resolve_libraries_path()) {
        red_debug_log("InitializeSdk redistributable_path=" + libraries_path_);
        init_status_ = InitializeSdk(libraries_path_.c_str(), OPTION_RED_NONE);
        if (init_status_ != ISInitializeOK) {
            throw std::runtime_error(
                "Failed to initialize RED SDK from redistributable path '" + libraries_path_
                + "' init_status=" + std::to_string(static_cast<int>(init_status_)) + "."
            );
        }
        red_debug_log("InitializeSdk status=" + std::to_string(static_cast<int>(init_status_)));
    }

    ~ScopedSdkSession() {
        red_debug_log("FinalizeSdk");
        FinalizeSdk();
    }

private:
    std::string libraries_path_;
    InitializeStatus init_status_ = ISLibraryNotLoaded;
};

class AlignedDecodeBuffer {
public:
    explicit AlignedDecodeBuffer(std::size_t size_bytes) : size_bytes_(size_bytes) {
        raw_buffer_ = static_cast<unsigned char *>(std::malloc(size_bytes_ + 15U));
        if (raw_buffer_ == nullptr) {
            throw std::runtime_error("Failed to allocate RED decode buffer.");
        }
        std::uintptr_t ptr = reinterpret_cast<std::uintptr_t>(raw_buffer_);
        const std::size_t adjustment = (ptr % 16U) == 0U ? 0U : 16U - (ptr % 16U);
        aligned_buffer_ = raw_buffer_ + adjustment;
        std::memset(aligned_buffer_, 0, size_bytes_);
    }

    ~AlignedDecodeBuffer() {
        std::free(raw_buffer_);
    }

    void *data() const {
        return aligned_buffer_;
    }

    std::size_t size_bytes() const {
        return size_bytes_;
    }

private:
    unsigned char *raw_buffer_ = nullptr;
    unsigned char *aligned_buffer_ = nullptr;
    std::size_t size_bytes_ = 0U;
};

py::dict clip_metadata_to_dict(const Clip &clip) {
    py::dict out;
    for (std::size_t index = 0; index < clip.MetadataCount(); ++index) {
        const std::string key = clip.MetadataItemKey(index);
        switch (clip.MetadataItemType(index)) {
            case MetadataTypeInt:
                out[py::str(key)] = clip.MetadataItemAsInt(index);
                break;
            case MetadataTypeFloat:
                out[py::str(key)] = clip.MetadataItemAsFloat(index);
                break;
            default:
                out[py::str(key)] = clip.MetadataItemAsString(index);
                break;
        }
    }
    return out;
}

py::object optional_clip_float(const Clip &clip, const char *metadata_key, double scale = 1.0) {
    if (!clip.MetadataExists(metadata_key)) {
        return py::none();
    }
    return py::float_(static_cast<double>(clip.MetadataItemAsFloat(metadata_key)) * scale);
}

void require_clip_loaded(const Clip &clip, const std::string &path) {
    red_debug_log("Clip status path=" + path + " status=" + std::to_string(static_cast<int>(clip.Status())));
    if (clip.Status() != LSClipLoaded) {
        throw std::runtime_error(
            "Failed to load R3D clip via RED SDK: " + path + " clip_status=" + std::to_string(static_cast<int>(clip.Status()))
        );
    }
}

struct DecodeAttemptResult {
    VideoDecodeMode mode;
    const char *mode_name;
    DecodeStatus status = DSDecodeOK;
};

std::string describe_attempts(const std::vector<DecodeAttemptResult> &attempts) {
    std::ostringstream stream;
    for (std::size_t index = 0; index < attempts.size(); ++index) {
        if (index > 0U) {
            stream << ", ";
        }
        stream << attempts[index].mode_name << "=" << static_cast<int>(attempts[index].status);
    }
    return stream.str();
}

py::array_t<float> decode_to_rgb_float32(
    const Clip &clip,
    const std::string &path,
    int frame_index,
    bool half_res,
    const std::string &colorspace,
    const std::string &gamma
) {
    if (frame_index < 0 || static_cast<std::size_t>(frame_index) >= clip.VideoFrameCount()) {
        throw std::runtime_error("Frame index out of range for RED clip decode.");
    }

    const std::size_t width = (clip.Width() + 1U) / 2U;
    const std::size_t height = (clip.Height() + 1U) / 2U;
    const std::size_t pixels = width * height;
    const std::size_t buffer_bytes = pixels * 3U * sizeof(std::uint16_t);
    AlignedDecodeBuffer buffer(buffer_bytes);
    std::ostringstream context;
    context
        << "clip_path=" << path
        << " frame_index=" << frame_index
        << " clip_status=" << static_cast<int>(clip.Status())
        << " full_width=" << clip.Width()
        << " full_height=" << clip.Height()
        << " decode_width=" << width
        << " decode_height=" << height
        << " decode_pixels=" << pixels
        << " buffer_bytes=" << buffer_bytes
        << " half_res=" << bool_string(half_res)
        << " colorspace=" << colorspace
        << " gamma=" << gamma
        << " total_frames=" << clip.VideoFrameCount();
    red_debug_log("Decode begin " + context.str());

    std::vector<DecodeAttemptResult> attempts;
    const std::vector<DecodeAttemptResult> candidates = {
        {DECODE_HALF_RES_GOOD, "DECODE_HALF_RES_GOOD", DSDecodeOK},
        {DECODE_HALF_RES_PREMIUM, "DECODE_HALF_RES_PREMIUM", DSDecodeOK},
    };

    DecodeStatus final_status = DSDecodeOK;
    bool decoded = false;
    for (const DecodeAttemptResult &candidate : candidates) {
        VideoDecodeJob job{};
        job.Mode = candidate.mode;
        job.PixelType = PixelType_16Bit_RGB_Planar;
        job.OutputBuffer = buffer.data();
        job.OutputBufferSize = buffer.size_bytes();
        Metadata frame_metadata;
        job.OutputFrameMetadata = &frame_metadata;

        red_debug_log(
            "Decode attempt mode=" + std::string(candidate.mode_name)
            + " pixel_type=PixelType_16Bit_RGB_Planar output_buffer_size=" + std::to_string(job.OutputBufferSize)
        );
        final_status = clip.DecodeVideoFrame(static_cast<std::size_t>(frame_index), job);
        attempts.push_back({candidate.mode, candidate.mode_name, final_status});
        red_debug_log(
            "Decode attempt result mode=" + std::string(candidate.mode_name)
            + " status=" + std::to_string(static_cast<int>(final_status))
        );
        if (final_status == DSDecodeOK) {
            decoded = true;
            break;
        }
    }

    if (!decoded) {
        throw std::runtime_error(
            "RED SDK frame decode failed. "
            + context.str()
            + " decode_attempts=[" + describe_attempts(attempts) + "]"
        );
    }

    const auto *planar = reinterpret_cast<const std::uint16_t *>(buffer.data());
    const auto *red = planar;
    const auto *green = planar + pixels;
    const auto *blue = planar + (pixels * 2U);

    py::array_t<float> array(
        {static_cast<py::ssize_t>(height), static_cast<py::ssize_t>(width), static_cast<py::ssize_t>(3)}
    );
    auto out = array.mutable_unchecked<3>();
    constexpr float scale = 1.0f / 65535.0f;
    for (std::size_t y = 0; y < height; ++y) {
        for (std::size_t x = 0; x < width; ++x) {
            const std::size_t offset = y * width + x;
            out(y, x, 0) = static_cast<float>(red[offset]) * scale;
            out(y, x, 1) = static_cast<float>(green[offset]) * scale;
            out(y, x, 2) = static_cast<float>(blue[offset]) * scale;
        }
    }
    return array;
}

#endif

}  // namespace

py::dict read_metadata(const std::string &path) {
#ifndef R3DMATCH_RED_SDK_ENABLED
    throw std::runtime_error(unavailable_message());
#else
    ScopedSdkSession sdk_session;
    Clip clip(path.c_str());
    require_clip_loaded(clip, path);

    py::dict payload;
    payload["source_path"] = path;
    payload["original_filename"] = basename_from_path(path);
    payload["width"] = static_cast<int>(clip.Width());
    payload["height"] = static_cast<int>(clip.Height());
    payload["total_frames"] = static_cast<int>(clip.VideoFrameCount());
    payload["fps"] = clip.VideoAudioFramerate();
    payload["iso"] = optional_clip_float(clip, RMD_ISO);
    payload["shutter_seconds"] = py::none();
    payload["aperture_t_stop"] = optional_clip_float(clip, RMD_LENS_APERTURE, 0.1);

    ImageProcessingSettings settings;
    clip.GetDefaultImageProcessingSettings(settings);
    payload["color_space"] = "REDWideGamutRGB";
    payload["gamma_curve"] = "Log3G10";
    py::dict extra_metadata = clip_metadata_to_dict(clip);
    extra_metadata["default_color_space_id"] = static_cast<int>(settings.ColorSpace);
    extra_metadata["default_gamma_curve_id"] = static_cast<int>(settings.GammaCurve);
    payload["extra_metadata"] = extra_metadata;
    return payload;
#endif
}

py::array_t<float> decode_frame(
    const std::string &path,
    int frame_index,
    bool half_res,
    const std::string &colorspace,
    const std::string &gamma
) {
#ifndef R3DMATCH_RED_SDK_ENABLED
    throw std::runtime_error(unavailable_message());
#else
    require_supported_decode_settings(half_res, colorspace, gamma);
    ScopedSdkSession sdk_session;
    Clip clip(path.c_str());
    require_clip_loaded(clip, path);
    try {
        return decode_to_rgb_float32(clip, path, frame_index, half_res, colorspace, gamma);
    } catch (const std::exception &exc) {
        throw std::runtime_error(
            "RED SDK decode failed for clip '" + path + "' frame " + std::to_string(frame_index) + ": " + exc.what()
        );
    }
#endif
}

PYBIND11_MODULE(_red_sdk_bridge, m) {
    m.doc() = "R3DMatch RED SDK bridge";
    m.def("sdk_available", &sdk_available);
    m.def("unavailable_message", &unavailable_message);
    m.def("read_metadata", &read_metadata, py::arg("path"));
    m.def(
        "decode_frame",
        &decode_frame,
        py::arg("path"),
        py::arg("frame_index") = 0,
        py::arg("half_res") = true,
        py::arg("colorspace") = "REDWideGamutRGB",
        py::arg("gamma") = "Log3G10"
    );
}
