#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <stdint.h>

#include "R3DSDK.h"
#include "R3DSDKMetadata.h"

namespace py = pybind11;
using namespace R3DSDK;

namespace {

std::string getenv_or_empty(const char *name) {
    const char *value = std::getenv(name);
    return value == nullptr ? std::string() : std::string(value);
}

void *aligned_malloc_16(size_t &adjustment, size_t size) {
    unsigned char *buffer = static_cast<unsigned char *>(std::malloc(size + 15U));
    if (!buffer) {
        return nullptr;
    }
    adjustment = 0U;
    uintptr_t ptr = reinterpret_cast<uintptr_t>(buffer);
    if ((ptr % 16U) == 0U) {
        return buffer;
    }
    adjustment = 16U - (ptr % 16U);
    return buffer + adjustment;
}

VideoDecodeMode decode_mode_from_string(const std::string &mode) {
    if (mode.empty() || mode == "full-premium") {
        return DECODE_FULL_RES_PREMIUM;
    }
    if (mode == "half-premium") {
        return DECODE_HALF_RES_PREMIUM;
    }
    if (mode == "half-good") {
        return DECODE_HALF_RES_GOOD;
    }
    throw std::runtime_error("Unsupported RED decode mode: " + mode);
}

std::pair<size_t, size_t> decoded_dimensions_for_mode(
    const std::string &mode,
    size_t full_width,
    size_t full_height
) {
    if (mode.empty() || mode == "full-premium") {
        return {full_width, full_height};
    }
    if (mode == "half-premium" || mode == "half-good") {
        return {(full_width + 1U) / 2U, (full_height + 1U) / 2U};
    }
    throw std::runtime_error("Unsupported RED decode mode: " + mode);
}

py::dict metadata_to_dict(const Metadata &metadata) {
    py::dict out;
    for (size_t i = 0; i < metadata.MetadataCount(); ++i) {
        const std::string key = metadata.MetadataItemKey(i);
        switch (metadata.MetadataItemType(i)) {
            case MetadataTypeInt:
                out[py::str(key)] = metadata.MetadataItemAsInt(i);
                break;
            case MetadataTypeFloat:
                out[py::str(key)] = metadata.MetadataItemAsFloat(i);
                break;
            default:
                out[py::str(key)] = metadata.MetadataItemAsString(i);
                break;
        }
    }
    return out;
}

py::dict clip_metadata_to_dict(const Clip &clip) {
    py::dict out;
    for (size_t i = 0; i < clip.MetadataCount(); ++i) {
        const std::string key = clip.MetadataItemKey(i);
        switch (clip.MetadataItemType(i)) {
            case MetadataTypeInt:
                out[py::str(key)] = clip.MetadataItemAsInt(i);
                break;
            case MetadataTypeFloat:
                out[py::str(key)] = clip.MetadataItemAsFloat(i);
                break;
            default:
                out[py::str(key)] = clip.MetadataItemAsString(i);
                break;
        }
    }
    return out;
}

}  // namespace

struct RedSdkConfig {
    std::string sdk_root;
    std::string libraries_path;
    bool use_gpu_decoder = false;
    std::string decode_mode = "full-premium";
};

class RedDecoderBackend {
public:
    explicit RedDecoderBackend(RedSdkConfig config) : config_(std::move(config)) {
        const std::string libs = resolve_libraries_path();
        init_status_ = InitializeSdk(libs.c_str(), OPTION_RED_NONE);
        initialized_ = init_status_ == ISInitializeOK;
    }

    ~RedDecoderBackend() {
        FinalizeSdk();
    }

    bool is_available() const {
        return initialized_;
    }

    py::dict sdk_diagnostics() const {
        py::dict diagnostics;
        diagnostics["backend"] = "red-sdk";
        diagnostics["available"] = initialized_;
        diagnostics["sdk_root"] = config_.sdk_root;
        diagnostics["libraries_path"] = resolve_libraries_path();
        diagnostics["sdk_version"] = GetSdkVersion();
        diagnostics["decode_mode"] = config_.decode_mode;
        diagnostics["message"] = initialized_
            ? "RED SDK initialized successfully"
            : "Failed to initialize RED SDK. Check RED SDK paths and runtime libraries.";
        diagnostics["init_status"] = static_cast<int>(init_status_);
        return diagnostics;
    }

    py::dict inspect_clip(const std::string &source_path) const {
        Clip clip(source_path.c_str());
        if (clip.Status() != LSClipLoaded) {
            throw std::runtime_error("Failed to load clip via RED SDK");
        }
        py::dict out;
        out["clip_id"] = py::str(source_path.substr(source_path.find_last_of("/\\") + 1));
        out["source_path"] = source_path;
        out["fps"] = clip.VideoAudioFramerate();
        out["width"] = static_cast<int>(clip.Width());
        out["height"] = static_cast<int>(clip.Height());
        out["total_frames"] = static_cast<int>(clip.VideoFrameCount());
        out["camera_model"] = "pinhole";
        ImageProcessingSettings settings;
        clip.GetDefaultImageProcessingSettings(settings);

        py::dict color_info;
        color_info["color_space"] = static_cast<int>(settings.ColorSpace);
        color_info["gamma_curve"] = static_cast<int>(settings.GammaCurve);
        color_info["iso"] = clip.MetadataExists(RMD_ISO) ? py::cast(clip.MetadataItemAsFloat(RMD_ISO)) : py::none();
        color_info["raw_bit_depth"] = 16;

        py::dict lens_info;
        lens_info["manufacturer"] = clip.MetadataExists(RMD_LENS_BRAND) ? clip.MetadataItemAsString(RMD_LENS_BRAND) : "";
        lens_info["model"] = clip.MetadataExists(RMD_LENS_NAME) ? clip.MetadataItemAsString(RMD_LENS_NAME) : "";
        lens_info["focal_length_mm"] = clip.MetadataExists(RMD_LENS_FOCAL_LENGTH) ? py::cast(clip.MetadataItemAsFloat(RMD_LENS_FOCAL_LENGTH)) : py::none();
        lens_info["aperture_t_stop"] = clip.MetadataExists(RMD_LENS_APERTURE) ? py::cast(clip.MetadataItemAsFloat(RMD_LENS_APERTURE) / 10.0f) : py::none();
        lens_info["focus_distance_m"] = clip.MetadataExists(RMD_LENS_FOCUS_DISTANCE) ? py::cast(clip.MetadataItemAsFloat(RMD_LENS_FOCUS_DISTANCE) / 1000.0f) : py::none();
        lens_info["additional"] = py::dict();

        py::dict extra_metadata = clip_metadata_to_dict(clip);
        color_info["additional"] = extra_metadata;

        out["color_info"] = color_info;
        out["lens_info"] = lens_info;
        out["reel_id"] = clip.MetadataExists(RMD_REEL_ID) ? py::cast(clip.MetadataItemAsString(RMD_REEL_ID)) : py::none();
        out["source_identifier"] = clip.MetadataExists(RMD_CAMERA_PIN) ? py::cast(clip.MetadataItemAsString(RMD_CAMERA_PIN)) : py::none();
        out["extra_metadata"] = extra_metadata;
        return out;
    }

    py::dict list_frames(const std::string &source_path) const {
        Clip clip(source_path.c_str());
        if (clip.Status() != LSClipLoaded) {
            throw std::runtime_error("Failed to load clip via RED SDK");
        }
        py::list frames;
        const double fps = clip.VideoAudioFramerate();
        for (size_t i = 0; i < clip.VideoFrameCount(); ++i) {
            py::dict frame;
            frame["frame_index"] = static_cast<int>(i);
            frame["timestamp_seconds"] = fps > 0.0 ? static_cast<double>(i) / fps : 0.0;
            frame["timecode"] = clip.AbsoluteTimecode(i);
            frames.append(frame);
        }
        py::dict out;
        out["frame_count"] = static_cast<int>(clip.VideoFrameCount());
        out["frames"] = frames;
        return out;
    }

    py::dict decode_frame(const std::string &source_path, size_t frame_index) const {
        Clip clip(source_path.c_str());
        if (clip.Status() != LSClipLoaded) {
            throw std::runtime_error("Failed to load clip via RED SDK");
        }
        const size_t full_width = clip.Width();
        const size_t full_height = clip.Height();
        const auto decoded_dims = decoded_dimensions_for_mode(config_.decode_mode, full_width, full_height);
        const size_t width = decoded_dims.first;
        const size_t height = decoded_dims.second;
        const size_t mem_needed = width * height * 3U * 2U;
        size_t adjustment = 0U;
        unsigned char *buffer = static_cast<unsigned char *>(aligned_malloc_16(adjustment, mem_needed));
        if (buffer == nullptr) {
            throw std::runtime_error("Failed to allocate RED decode buffer");
        }

        VideoDecodeJob job;
        job.Mode = decode_mode_from_string(config_.decode_mode);
        job.PixelType = PixelType_16Bit_RGB_Planar;
        job.OutputBuffer = buffer;
        job.OutputBufferSize = mem_needed;
        Metadata frame_metadata;
        job.OutputFrameMetadata = &frame_metadata;

        const DecodeStatus status = clip.DecodeVideoFrame(frame_index, job);
        if (status != DSDecodeOK) {
            std::free(buffer - adjustment);
            throw std::runtime_error("RED SDK frame decode failed");
        }

        py::bytes raw(reinterpret_cast<char *>(buffer), mem_needed);
        std::free(buffer - adjustment);

        py::dict frame_payload;
        frame_payload["width"] = static_cast<int>(width);
        frame_payload["height"] = static_cast<int>(height);
        frame_payload["channels"] = 3;
        frame_payload["bit_depth"] = 16;
        frame_payload["layout"] = "C,H,W planar RGB";
        frame_payload["data"] = raw;

        py::dict out;
        out["frame"] = frame_payload;
        out["frame_metadata"] = metadata_to_dict(frame_metadata);
        return out;
    }

private:
    std::string resolve_libraries_path() const {
        if (!config_.libraries_path.empty()) {
            return config_.libraries_path;
        }
        const std::string env_lib = getenv_or_empty("RED_SDK_REDISTRIBUTABLE_DIR");
        if (!env_lib.empty()) {
            return env_lib;
        }
#ifdef R3DSPLAT_RED_SDK_REDIS_DIR
        return R3DSPLAT_RED_SDK_REDIS_DIR;
#else
        return ".";
#endif
    }

    RedSdkConfig config_;
    InitializeStatus init_status_ = ISLibraryNotLoaded;
    bool initialized_ = false;
};

PYBIND11_MODULE(_r3d_native, m) {
py::class_<RedSdkConfig>(m, "RedSdkConfig")
    .def(py::init<const std::string&, const std::string&, bool, const std::string&>(),
         py::arg("sdk_root") = "",
         py::arg("libraries_path") = "",
         py::arg("use_gpu_decoder") = false,
         py::arg("decode_mode") = "full-premium")
    .def_readwrite("sdk_root", &RedSdkConfig::sdk_root)
    .def_readwrite("libraries_path", &RedSdkConfig::libraries_path)
    .def_readwrite("use_gpu_decoder", &RedSdkConfig::use_gpu_decoder)
    .def_readwrite("decode_mode", &RedSdkConfig::decode_mode);

    py::class_<RedDecoderBackend>(m, "RedDecoderBackend")
        .def(py::init<RedSdkConfig>(), py::arg("config") = RedSdkConfig{})
        .def("is_available", &RedDecoderBackend::is_available)
        .def("sdk_diagnostics", &RedDecoderBackend::sdk_diagnostics)
        .def("inspect_clip", &RedDecoderBackend::inspect_clip)
        .def("list_frames", &RedDecoderBackend::list_frames)
        .def("decode_frame", &RedDecoderBackend::decode_frame);
}
