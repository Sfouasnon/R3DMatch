/* R3D SDK library version 9.2 header file. Do *NOT* use this
   header file with any other version of the R3D SDK library!
   
   This header file and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All
   rights reserved. Redistribution of this header is prohibited!
*/

#ifndef R3DSDK_H
#define R3DSDK_H

#include <string>

#include "R3DSDKDefinitions.h"
#include "R3DSDKMetadata.h"

namespace R3DSDK {

// Bit flags that can be passed to InitializeSdk() through the optional_components parameter
#define OPTION_RED_NONE				0x00
#define OPTION_RED_CUDA				0x01
#define OPTION_RED_OPENCL			0x02
#define OPTION_RED_DECODER			0x04	// exclusive, do not combine with OPTION_RED_CUDA, OPTION_RED_OPENCL or OPTION_RED_METAL flags
#define OPTION_RED_METAL        	0x08	// macOS only
#define OPTION_DELAY_GPU_COMPILE	0x10	// OpenCL only: don't pre-compile kernels in REDCL::checkCompatibility(), instead compile them when needed

    
// This must be called one time before calling any other functions or
// constructing any classes. Do not call this for each single thread.
// Static initializers must have run before calling this.
// UTF-8 pathToDynamicLibraries must point to the folder where the
// dynamic libraries can be found, it cannot be NULL or an empty string.
InitializeStatus InitializeSdk(const char * pathToDynamicLibraries, unsigned int optional_components);

// This must be called anytime InitializeSdk() has been called,
// whether or not InitializeSdk() succeeded!
void FinalizeSdk();

// Returns version information for the static/API version as well as any dynamic
// libraries succesfully loaded. Should be called after InitializedSdk(), but will
// return useful information even if initialization failed. If possible log the
// output of this function to a log file to help with customer issue debugging.
const char * GetSdkVersion();

// Identify the type of file / clip in a quick way without full parsing. This
// is particularly useful to determine if a clip can have multiple file parts
// (FileId_R3D) that should be presented as a single clip or if each file is
// its own individual clip. A positive identification of a format does not
// guarantee the SDK can successfully open & decode the given file / clip.
enum FileId
{
	FileId_Unknown  = 0,	// unknown/can't open or read
	FileId_R3D      = 1,	// original .R3D with multi-file/spanning clips (A001_C001_1231XY_001...999.R3D, A001_C002_1231XY_001...999.R3D, etc.)
	FileId_NEV_NRAW = 3,	// Nikon .NEV N-RAW with single file per clip (DSC_0001.NEV, DSC_0002.NEV, etc.)
	FileId_R3D_NE   = 4		// .R3D with single file per clip (DSC_0001.R3D, DSC_0002.R3D, etc.)
};

int IdentifyFile(const char * utf8Path);
    
// Class for loading a clip, retrieving information and extracting images.
// This class is thread-safe as long as no call to LoadFrom() or Close()
// is in progress.
class Clip
{
public:
	// ******************************
	// CLIP MANAGEMENT
	// ******************************

	// Destructor will call Close() to close any open files
	~Clip();

	// First constructor form. Use this if you do not want to load a clip
	// when constructing this class. Use LoadFrom() to load a clip before
	// calling any other functions.
	Clip();

	// Second constructor form. Use this if you want to load a clip from a UTF-8
	// path when you construct the class. Load status can be checked through Status().
	Clip(const char * pathToFile);

	// Get clip load status after having used the second constructor form
	// or after a call to LoadFrom().
	LoadStatus Status() const;

	// Load the R3D file from the given UTF-8 path. If there are any other R3D
	// files that belong to the same clip these will get loaded as well.
	// If a clip was already open it will be closed before opening the
	// one specified in the pathToFile parameter.
	LoadStatus LoadFrom(const char * pathToFile);

	// Closes any open files (class destructor will also call this function).
	void Close();

	// Closes all handles for all R3D parts belonging to the clip. Handles
	// will be re-opened on a per R3D part basis as needed automatically.
	void CloseFileHandles();

	// Indicates if RMD sidecar file was found when clip was loaded
	bool RmdSidecarPresent() const;

	// Retrieve the UTF-8 path for the RMD sidecar file the R3D SDK will look
	// for when trying to load or update an RMD file or create a new one.
	// You will receive a valid path even if no RMD file is present
	// Returns null if no clip has been (succesfully) loaded.
	const char * GetRmdPath() const;
	
	// Checks frame CRC if present. For HDRx both tracks are checked. CRC specific return values:
	// 
	// DSDecodeOK             : CRC present and matched
	// DSDecodeFailed         : CRC present and check failed
	// DSUnsupportedClipFormat: clip/frame does not have CRC
	DecodeStatus CheckFrame(size_t videoFrameNo) const;

	// ******************************
	// SAVING CLIPS
	// ******************************
	// The callback will be called whenever a frame has been added, the
	// operation has been completed or an error has occurred. The last callback
	// you will get is either the done status or an error. If 'status' does
	// not equal CSFrameAdded the process is done or an error was encountered.
	// 
	// Return 'true' from your callback if you want the trim to continue
	// Return 'false' if you want to abort the trim
	typedef bool (*TrimCallback)(CreateStatus status, void * privateData, size_t frameNoDone, size_t lastFrameToDo);

	// Create a new clip from another one with a specific in and out point.
	// 'destination' is an existing folder where the output files will be created
	// 'destination' is a UTF-8 path
	// 'destination' must not already contain the same clip
	//
	// This is an asynchronous call, it will return immediately. The 'destination'
	// parameter can be free'd or deleted as soon as this call returns. The
	// 'source' clip must remain around for the duration of this function.
	// The function returns CSStarted if the trim is under way and you will
	// receive at least one more callback when this value is returned.
	static CreateStatus CreateTrimFrom(const Clip & source, const char * destination, size_t startFrameNo, size_t endFrameNo, bool includeAudio, void * privateData, TrimCallback callback);

	// Create single frame clip from another one. 'destination' is a full UTF-8 path
	// including filename (unlike CreateTrimFrom above). The extension '.R3D'
	// must be at the end and it must be in capitals! The last character in the
	// filename part of the path CANNOT be a digit! This call returns immediately.
	static CreateStatus CreateSnapshotFrom(const Clip & source, const char * destination, size_t frameNo);

	// If you try to include a frame (either through CreateTrimFrom or CreateSnapshotFrom)
	// that is a dropped frame you will receive CSFailedToGetSourceFrame return code.
	// In this case CreateSnapshotFrom will not have created an output file at all
	// In this case CreateTrimFrom will have created a clip up to the dropped frame.

	// ******************************
	// SYNCHRONOUS SOFTWARE DECODING (MAIN TRACK)
	// ******************************

	// Decode the given video frame with the supplied decode settings,
	// output buffer and image processing settings (through decodeJob)
	DecodeStatus DecodeVideoFrame(size_t videoFrameNo, const VideoDecodeJob & decodeJob) const;

	// Turn off, or back on, Extended Highlights decoding for V-RAPTOR [X] clips shot in this mode.
	bool ExtendedHighlightsEnabled() const;
	bool SetExtendedHighlights(bool enabled);	// returns true if success

	// Decode the given audio block into the supplied buffer. This audio block has 24- or 32-bit, see
	// the RMD_SAMPLE_SIZE metadata, Big Endian audio. For 24-bit audio it will be MSB aligned in a
	// 32-bit word. In that case the lower 8-bits are set to zero. If there is more than one audio
	// channel (see AudioChannelCount() API below) the samples will be interleaved.
	//
	// Clips with floating point audio will automatically be converted to integer with the gains set
	// to the values returned by the GetFloatAudioDefaultConversionGain() API.
	//
	// outputBuffer must point to a 512-byte aligned buffer that is big enough to hold the requested
	// audio block (see AudioBlockCountAndSize() API below). The size of the memory block must be set
	// in bufferSize which will be updated by the actual number of bytes written into the audio buffer
	// which will always be a multiple of 4 * AudioChannelCount().
	//
	// The function will fail if no clip is open, no audio is present, audioBlockNo is out of range or
	// outputBuffer and bufferSize are NULL or not aligned properly. This function is *NOT* thread-safe!
	DecodeStatus DecodeAudioBlock(size_t audioBlockNo, void * outputBuffer, size_t * bufferSize) const;

	// Decode raw audio samples from the clip. startSample and numberOfSamples are expressed per channel,
	// do not multiply them by the channel count! The audio returned has 24- or 32-bit, see the
	// RMD_SAMPLE_SIZE metadata, Big Endian audio. For 24-bit audio it will be MSB aligned in a 32-bit
	// word. In that case the lower 8-bits are set to zero. If there is more than one audio channel
	// (see AudioChannelCount() API below) the samples will be interleaved.
	//
	// numberOfSamples will be updated to reflect how many samples were actually written into the
	// output buffer. Normally this is the same as the number of samples requested, unless more
	// samples were requested than exist in the clip
	//
	// Clips with floating point audio will automatically be converted to integer with the gains set
	// the values returned by the GetFloatAudioDefaultConversionGain() API.
	//
	// outputBuffer must point to a 512-byte aligned buffer that is big enough to hold the requested
	// audio samples. The size of the memory block must be set in bufferSize, which will be updated by
	// the actual number of bytes written into the audio buffer, which will always be a multiple of
	// 4 * AudioChannelCount().
	//
	// The function will fail if no clip is open, no audio is present, startSample is out of range or
	// outputBuffer and bufferSize are NULL or not aligned properly. On failure numberOfSamples will
	// be set to zero. This function is *NOT* thread-safe!
	DecodeStatus DecodeAudio(unsigned long long startSample, size_t * numberOfSamples, void * outputBuffer, size_t bufferSize) const;

	// Overloaded versions of DecodeAudioBlock() & DecodeAudio() APIs. These versions allow setting
	// the float-to-integer conversion gain to be used per-channel.
	// 
	// channelGainDecibels may be NULL, in which case the values returned by the
	// GetFloatAudioDefaultConversionGain() API are used. When channelGainDecibels is not NULL the
	// pointer to array MUST have at least AudioChannelCount() entries!
	// 
	// Channel gain values below -50 dB apply 0 gain (mutes the channel) and values above +50dB
	// are clamped to 50 to have a useful User Interface range.
	//
	// If these functions are called for a clip without floating point audio DSUnsupportedClipFormat
	// is returned as the SDK does not support decoding integer audio with gains applied.
	DecodeStatus DecodeAudioBlock(size_t audioBlockNo, void * outputBuffer, size_t * bufferSize, const int * channelGainDecibels) const;
	DecodeStatus DecodeAudio(unsigned long long startSample, size_t * numberOfSamples, void * outputBuffer, size_t bufferSize, const int * channelGainDecibels) const;

	// Auadio decode APIs for use with native float audio clips to decode the floating point audio as-is.
	// These work exactly the same as the integer versions with the exception that they will fail if
	// the clip does not have floating point audio as the SDK does not support integer-to-float conversion.
	DecodeStatus DecodeFloatAudioBlock(size_t audioBlockNo, void * outputBuffer, size_t * bufferSize) const;
	DecodeStatus DecodeFloatAudio(unsigned long long startSample, size_t * numberOfSamples, void * outputBuffer, size_t bufferSize) const;

	// Return the per-channel audio peak amplitude for clips with floating point audio. The function
	// will fail and return 0.0 if no clip is open, the clip has no floating point audio or
	// channelNo is out of bounds.
	float GetFloatAudioPeakAmplitude(size_t channelNo) const;

	// Return the pre-calculated per-chnnal audio gain in dB that can be used to convert floating
	// point audio to integer or could be used as a default in a User Interface. This function will
	// fail and return 0dB if no clip is open, the clip has no floating point audio or channelNo
	// is out of bounds.
	int GetFloatAudioDefaultConversionGain(size_t channelNo) const;
	
	// ******************************
	// MULTI-TRACK VIDEO DECODING
	// Can be used to do manual HDR blending or decode a single track
	// ******************************
	
	// Returns number of video tracks in clip. This will always return 1 for non-HDRx clips
	size_t VideoTrackCount() const;

	// Decode the given video frame on the given track with the supplied decode
	// settings, output buffer and image processing settings (through decodeJob)
	DecodeStatus VideoTrackDecodeFrame(size_t videoTrackNo, size_t videoFrameNo, const VideoDecodeJob & decodeJob) const;

	// ******************************
	// ASYNC IMU API FOR V-RAPTOR RECORDINGS
	// see RMD_FRAME_TIMESTAMP metadata and Clip::GetFrameMetadata() API as well
	// ******************************
	size_t IMUSampleCount() const;

	// a sampleCount of 0 = provides all samples from startSample till end
	bool DecodeIMUSamples(IMUSample * outSamples, size_t startSample = 0, size_t sampleCount = 0);

	// ******************************
	// XMP RMD FUNCTIONS
	// ******************************

	// Adds or updates XMP data inside the RMD sidecar file. If this file does
	// not exist it will be created. In the latter case it probably makes sense
	// to call CreateOrUpdateRmd() below as well to add image processing settings.
	// The 'xmpString' parameter must be a std::string UTF-8 encoded.
	// This function returns true if XMP was added. It returns false if an empty
	// input string was supplied or the add/update failed for other reasons.
	bool CreateOrUpdateRmd(const std::string xmpInput) const;

	// Fills 'xmpOutput' with the XMP data (in UTF-8 as a std::string) from the
	// RMD sidecar file if present. If the file is not present or does not contain
	// any XMP data then 'xmpOutput' will be untouched and false will be returned
	bool GetRmdXmp(std::string & xmpOutput) const;

	// ******************************
	// IMAGE PROCESSING SETTINGS
	// ******************************

	// Get either the clip image processing settings or overriding RMD sidecar metadata
	// (display this in your UI as the clip defaults!). Use GetClipImageProcessingSettings()
	// function below to always get the clip settings. PERFORMANCE WARNING: This function
	// will check for, and parse the RMD sidecar, EVERY time this function gets called!
	void GetDefaultImageProcessingSettings(ImageProcessingSettings & settingsToFill) const;

	// Get the image processing settings as they were set on camera when the clip was
	// recorded. Use this to offer "reset to clip/recorded metadata" in your application.
	void GetClipImageProcessingSettings(ImageProcessingSettings & settingsToFill) const;

	// Create or update existing RMD sidecar file. Returns true if success, false otherwise
	// The ImageProcessingSettings struct *must* have Version set to ColorVersion2 or above
	// for this call to succeed. ColorVersion1 is not supported within RMD sidecar files.
	// The RMD sidecar file is an XML based format that travels with the clips. In the future
	// RED cameras will create this file and third party's can add their own data.
	bool CreateOrUpdateRmd(const ImageProcessingSettings & settingsToWrite) const;

	// Get the HDRx processing settings from an existing RMD sidecar file
	// Returns blend settings OR track number to decode. If an error occurs, or no HDR
	// processing settings exist this function will return track 0 (A / main track)
	HdrMode GetRmdHdrProcessingSettings(HdrProcessingSettings & settingsToFill, size_t & trackNoToFill) const;

	// Create or update existing RMD sidecar file either a track number OR HDRx
	// processing settings. Returns true if success, false otherwise
	bool CreateOrUpdateRmd(HdrMode mode, const HdrProcessingSettings & settingsToWrite, size_t trackNoToWrite) const;

	// Whitebalance: convert a pixel area to Kelvin & Tint
	// End-user can pick a point on the image for a given frame. X and Y are in 0.0 - 1.0
	// range and go from upper left (0,0) to lower right (1,1). Function fails if x, y or
	// videoFrameNo are out of bounds, clip has not been loaded or frame is corrupt.
	// It will only update Kelvin & Tint for the supplied ImageProcessingSettings, nothing else
	DecodeStatus GetWhitebalance(ImageProcessingSettings & settingsToFill, size_t videoFrameNo, float x, float y) const;

	// Minimum color version of the ImageProcessingSettings structure required for the clip.
	// If the version is set lower in ImageProcessingSettings than the value returned by this
	// function it will be changed to the highest color version available. This function can
	// help to identify older footage that you may want to enable older image processing for
	// to match how the image used to look in an older SDK (ie, backwards compatibility).
	ColorVersion MinimumColorVersion() const;

	// Returns Default Color Version for clip. ColorVersion3 for clips recorded in-camera
	// with IPP2 color science, ColorVersion2 (Legacy) for all other clips.
	ColorVersion DefaultColorVersion() const;

	// True if 3D LUT was set by user in-camera when clip was recorded.
	bool Camera3DLutPresent() const;

	// Load 3D LUT (.cube only!) for use in IPP2 decoding. Load search sequence:
	// 	1) if filename only, try to load as sidecar file next to the clip.
	// 	2) try to load as a full path
	// 	3) try to load from current directory
	// If the 3D LUT is a sidecar file then it's important to only supply the filename and not
	// the full path so it can later be saved correctly to the sidecar RMD file when CreateOrUpdateRmd()
	// is called above. There is also a static version without sidecar behavior.
	// Supports .cube files with edge lengths between 17 and 65 inclusive
	Handle3DLut Load3DLut(const char * path) const;

	// ******************************
	// CLIP INFORMATION
	// ******************************

	// Get a unique 16-byte clip identifier. There is no guarantee that this
	// uuid follows the ITU-T Rec. X.667 / ISO/IEC 9834-8:2005 standards!
	// Returns false if no clip is loaded or uuid is NULL, otherwise returns
	// true. Output uuid buffer must be able to hold 16 bytes.
	bool Uuid(unsigned char * uuid) const;

	// Width of the clip, will return 0 if no clip is loaded.
	size_t Width() const;

	// Height of the clip, will return 0 if no clip is loaded.
	size_t Height() const;

	// Number of video frames in the clip, will return 0 if no clip is loaded.
	size_t VideoFrameCount() const;

	// Number of audio channels in the clip, will return 0 if no
	// clip is loaded or if the clip does not have any audio. To
	// get the channel map (which of the 4 channels was used),
	// query the RMD_CHANNEL_MASK metadata item.
	size_t AudioChannelCount() const;

	// Number of audio blocks and their maximum size in the clip
	// Will return 0 if no clip is loaded or if maximumSize is
	// NULL, otherwise returns number of audio blocks and sets
	// maximumSize to the maximum buffer size that will be
	// needed to process all audio blocks. This size will be a
	// multiple of 512 (guaranteed)
	size_t AudioBlockCountAndSize(size_t * maximumSize) const;

	// Number of audio samples *per channel* in the clip. All
	// channels have the exact same number of samples. Returns
	// 0 if an error is encountered or if the clip has no audio
	unsigned long long AudioSampleCount() const;

	// Framerate for the video & audio 'track(s)'. Numerator & denominator
	// can be found seperately in the metadata table. Will return 0.0 if no
	// clip is loaded.
	float VideoAudioFramerate() const;	

	// Framerate for the timecode 'track'. This will be running at half the
	// video & audio 'track' rate if these are running at over 30 fps. The
	// timecode functions below will alternate '.' and ':' in that case!
	// Will return 0.0 if no clip is loaded.
	float TimecodeFramerate() const;

	// Retrieve the clip's default timecode 'track' for a given video frame.
	// This default was set on the camera when the clip was recorded. The
	// returned value is valid until the next time this function is called
	// or either AbsoluteTimecode() or EdgeTimecode() depending on which
	// 'track' is the default. Will return a NULL pointer if no clip is loaded.
	// or if videoFrameNo is out of bounds.
	const char * Timecode(size_t videoFrameNo);

	// Retrieve the clip's absolute (time of day or external) timecode for
	// a given video frame. The returned value is valid until the next time
	// this function or Timecode() is called. Will return a NULL pointer if
	// no clip is loaded or if VideoFrameNo is out of bounds.
	const char * AbsoluteTimecode(size_t videoFrameNo);

	// Retrieve the clip's edge (aka run record) timecode for a given video
	// frame. The returned value is valid until the next time this function
	// or Timecode() is called. Will return a NULL pointer if no clip is
	// loaded or videoFrameNo is out of bounds.
	const char * EdgeTimecode(size_t videoFrameNo);
	
	// You can encounter two kinds of timecodes in an R3D file:
	//
	// 01:00:00:00 -> 01:00:00:01 etc. for clips running at 30 fps and below
	// 01.00.00.00 -> 01:00:00:00 etc. for clips running over 30 fps

	// Function to retrieve per-frame metadata. This is an expensive call as
	// the frame is loaded from the disk everytime this function is called!
	// All decoder interfaces (Software & GPU) can return per-frame
	// metadata as frames are decoded which incurs no extra disk I/O hit.
	// Recommended use for GetFrameMetadata() is when dumping per-frame metadata
	// for all (or a certain range of) frames without needing the decoded image.
	DecodeStatus GetFrameMetadata(Metadata & metadataToFill, size_t videoFrameNo) const;

	// External Metadata is metadata that was fed in to the camera during record
	// from some external source. This is only available through this API,
	// and the amount of metadata present can change from frame to frame.
	// Supply a pointer to externalHasChanged to see if any of the external
	// metadata has changed from the previous frame recorded.
	DecodeStatus GetFrameMetadata(Metadata & metadataToFill, size_t videoFrameNo, Metadata * externalMetadataToFill, bool * externalHasChanged) const;

	// Functions to retrieve clip (first frame) metadata through the metadata database
	size_t MetadataCount() const;								// number of metadate items in the store
	bool MetadataExists(const char * key) const;				// check if a metadata item exists in the store
	std::string MetadataItemKey(size_t index) const;			// get key for an index

	// keep in mind that there is no guarantee which metadata item will be at which
	// index. The index option is only there to allow enumeration through the database.
	unsigned int MetadataItemAsInt(const char * key) const;		// retrieve metadata item by key and return as int
	unsigned int MetadataItemAsInt(size_t index) const;			// retrieve metadata item by index and return as int
	std::string MetadataItemAsString(const char * key) const;	// retrieve metadata item by key and return as STL string
	std::string MetadataItemAsString(size_t index) const;		// retrieve metadata item by index and return as STL string
	float MetadataItemAsFloat(const char * key) const;			// retrieve metadata item by key and return as float
	float MetadataItemAsFloat(size_t index) const;				// retrieve metadata item by index and return as float
	
	MetadataType MetadataItemType(const char * key) const;		// retrieve metadata item type by key
	MetadataType MetadataItemType(size_t index) const;			// retrieve metadata item type by index

	bool ExternalETMetadataPresent() const;
	size_t ExternalETMetadataCount() const;
	const char * ExternalETMetadata(size_t index) const;

	// Functions to get still frame markers creating during record
	size_t MarkerCount() const;
	bool GetMarker(size_t markerNo, Marker & markerToFill) const;

	enum FileType
	{
		FileType_Invalid  = 0,	// fileIdx invalid in FileList() call
		
		// Files that are part of the clip itself, there's always at least 1 of these
		FileType_Clip     = 1,	// .R3D or .NEV
		
		// Optional sidecar / external files
		FileType_Metadata = 2,	// .RMD or .cdl
		FileType_LUT      = 3,	// .cube
		FileType_Audio    = 5	// .wav
	};

	// These APIs only return information about about the clip sidecar files with
	// the above extensions. Any other files in the same folder are ignored
	size_t FileListCount() const;
	FileType FileList(size_t fileIdx, std::string & pathOutput) const;

private:
	// no copying & assignment
	Clip(const Clip &) throw();
	Clip & operator= (const Clip &) throw();

	void * reserved;
};

// The asynchronous decompress job structure. This structure needs to be
// constructed to submit a decode job to AsyncDecoder::DecodeForGpuSdk().
// Every field must be set, with the exception of 'Callback' & 'PrivateData'.
struct AsyncDecompressJob
{
private:
	unsigned int reserved1;
	unsigned int reserved2;

public:
	// Set this flag to true if you want to abort processing this frame as soon as possible
	bool						AbortDecode;

	// Frame to decode for which clip
	R3DSDK::Clip *				Clip;

	size_t						VideoTrackNo;			// 0 = main (A) track, 1 = EPIC/Scarlet-X higlight protection track 2 (X track)

	size_t						VideoFrameNo;

	// item is the pointer to the original item as submitted in the Decode()
	typedef void (*AsyncDecompressCallback)(AsyncDecompressJob * item, DecodeStatus decodeStatus);

	// Callback to call when frame is available in the specified output buffer
	// You will receive a call for each frame submitted and these calls will
	// come in on seperate threads, potentially at the same time. THIS PARAMETER IS REQUIRED!
	// SPEND AS LITTLE TIME AS POSSIBLE IN THE CALLBACK TO NOT HOLD UP PROCESSING OTHER FRAMES!
	AsyncDecompressCallback		Callback;
	void *						PrivateData;			// available for your use as you see fit, R3D SDK will not touch this variable!

	// Resolution/speed to decode the image at. This will also 
	// influence how much memory is needed for the output buffer
	VideoDecodeMode				Mode;

	// Pointer to the buffer to store the image in. This cannot be
	// NULL otherwise the decode will fail. The buffer must be aligned
	// on a 16-byte boundary (see sample code in GPU extension).
	void *						OutputBuffer;

	// Size ouf the output buffer. If this value is too small the decode will fail.
	// see AsyncDecoder::GetSizeBufferNeeded()
	size_t						OutputBufferSize;
	
	// Optionally return the decoded frame's per-frame metadata
	Metadata *					OutputFrameMetadata;
	
	AsyncDecompressJob();
};

// Class for asynchronously decompressing a frame for further processing on the GPU.
class AsyncDecoder
{
public:
	AsyncDecoder();
	~AsyncDecoder();

	// Returns the number of threads the decoder will use if opened when
	// noOfThreads in Open() is set to 0 (the default).
	static size_t ThreadsAvailable();

	// Open the asynchronous decoder with the specified number of threads
	// If noOfThreads is not set, or set to 0, ThreadsAvailable() number
	// of threads will be used.
	void Open(size_t noOfThreads = 0U);
	
	// Close asynchronous decoder and clean up any threads & memory in use
	void Close();

	// Returns the size OutputBufferSize needs to be in the AsyncDecompressJob
	// so the appropriate size input buffer can be allocated before calling
	// the DecodeForGpuSdk() function. The following must be set on input:
	//  1) job.Clip must point to an open Clip and cannot be NULL
	//  2) job.Mode must be set appropriately
	// returns 0 if these input parameters are invalid or no clip is loaded
	static size_t GetSizeBufferNeeded(const AsyncDecompressJob & job);

	// Decompress a frame in to the supplied OutputBuffer for further
	// processing on the GPU. NOTE THIS IS AN ASYNCHRONOUS CALL!
	DecodeStatus DecodeForGpuSdk(AsyncDecompressJob & job);

private:
	// no copying & assignment
	AsyncDecoder(const AsyncDecoder &) throw();
	AsyncDecoder & operator= (const AsyncDecoder &) throw();
	
	void * reserved;
};

// Class for asynchronously processing a frame for later decompression
// and image processing through CUDA (Windows & Linux) or Metal (macOS).
// NOTE: 4GB minimum video RAM recommended when using GPU decode
class GpuDecoder
{
public:
	GpuDecoder();
	~GpuDecoder();

	// Open the asynchronous decoder.
	void Open();

	// Close asynchronous decoder and clean up any threads & memory in use
	void Close();

	// Determines if the current clip is supported for GPU decompression.
	// All clips except for ones recorded on RED ONE are supported.
	static DecodeStatus DecodeSupportedForClip(const Clip & clip);

	// Returns the size OutputBufferSize needs to be in the AsyncDecompressJob
	// so the appropriate size input buffer can be allocated before calling
	// the DecodeForGpuSdk() function. The following must be set on input:
	//  1) job.Clip must point to an open Clip and cannot be NULL
	//  2) job.Mode must be set appropriately
	// returns 0 if these input parameters are invalid or no clip is loaded
	static size_t GetSizeBufferNeeded(const AsyncDecompressJob & job);

	// Prepare a frame for decompression & further processing on the GPU.
	// NOTE THIS IS AN ASYNCHRONOUS CALL!
	DecodeStatus DecodeForGpuSdk(AsyncDecompressJob & job);

private:
	// no copying & assignment
	GpuDecoder(const GpuDecoder &) throw();
	GpuDecoder & operator= (const GpuDecoder &) throw();

	void * m;
};

// Load 3D LUT (.cube only!) for use in IPP2 decoding. Load search sequence:
// 	1) if filename only, try to load from clipPath if present
// 	2) try to load as a full path
// 	3) try to load from current directory
// If the 3D LUT is a sidecar file then it's important to call Clip::Load3DLut()
// version of this API instead! See description of that API to get more information.
// Supports .cube files with edge lengths between 17 and 65 inclusive
Handle3DLut Load3DLut(const char * lutPath, const char * clipPath = NULL);

// Unload 3D LUTs loaded through one of the Load3DLut() APIs. The supplied
// handle will be set to NULL for you. Can be safely called on R3D/RMD auto loaded 3D LUTs.
void Unload3DLut(Handle3DLut * handle);

// Get the path for the given 3D LUT. Returns NULL if handle is invalid.
// If this is a sidecar 3D LUT then only the filename will be returned.
const char * Get3DLutPath(Handle3DLut handle);

// Save IPP2 entire Output Transform to disk as a single .cube 3D LUT.
// The Output Transform takes the RWG/Log3G10 to the final output image via the user selected settings.
// Returns CSDone if succesful, otherwise returns error code. The following parameters are used:
// - utf8OutputPath  : UTF-8 path to folder where LUT will be created in. Existing LUT with same name will be overwritten!
// - edgeLength      : must be in range 17 -- 65 inclusive
// - colorSpace      : output color space to convert to from REDWideGamutRGB
// - gammaCurve      : output gamma to convert to from Log3G10
// - outputToneMap   : tone map to use
// - highlightRollOff: highlight roll off to use
// - hdrPeakNits     : only used when HDR2084 is selected
CreateStatus SaveRWGLog3G10ToOutputTransform3DLut(const char * utf8OutputPath, int edgeLength,
                                                  ImageColorSpace colorSpace, ImageGammaCurve gammaCurve, ToneMap outputToneMap, RollOff highlightRollOff, unsigned int hdrPeakNits);

// Save IPP2 entire Output Transform to memory as a 3D LUT.
// The Output Transform takes the RWG/Log3G10 to the final output image via the user selected settings.
// Returns false if an invalid parameter was supplied. The following parameters are used:
// - output3DLut     : memory must be allocated before calling function. Size must be at least:
//                     edgeLength * edgeLength * edgeLength * sizeof(float)
// - edgeLength      : must be in range 17 -- 65 inclusive
// - colorSpace      : output color space to convert to from REDWideGamutRGB
// - gammaCurve      : output gamma to convert to from Log3G10
// - outputToneMap   : tone map to use
// - highlightRollOff: highlight roll off to use
// - hdrPeakNits     : only used when HDR2084 is selected
bool CreateRWGLog3G10ToOutputTransform3DLut(float * output3DLut, int edgeLength,
                                            ImageColorSpace colorSpace, ImageGammaCurve gammaCurve, ToneMap outputToneMap, RollOff highlightRollOff, unsigned int hdrPeakNits);

// Save IPP2 Output Transform to memory as a 1D LUT for the Tone Map Curve followed by a 3D LUT.
// The Output Transform takes the RWG/Log3G10 to the final output image via the user selected settings.
// Returns false if an invalid parameter was supplied. The following parameters are used:
// - output3DLut     : memory must be allocated before calling function. Size must be at least:
//                       3 * edgeLength * edgeLength * edgeLength * sizeof(float)
// - edgeLength      : must be in range 17 -- 65 inclusive
// - output1DLut     : memory must be allocated before calling function. Size must be at least:
//                       lutSize * sizeof(float)
// - lutSize         : entries to create in 1D Tone Curve LUT. Must be in range 1024 -- 65536 inclusive.
// - lutScaleFactor  : factor to scale entries in 1D Tone Curve LUT by. Must be 1 or higher.
// - colorSpace      : output color space to convert to from REDWideGamutRGB
// - gammaCurve      : output gamma to convert to from Log3G10
// - outputToneMap   : tone map to use
// - highlightRollOff: highlight roll off to use
// - hdrPeakNits     : only used when HDR2084 is selected
bool CreateRWGLog3G10ToOutputTransformLuts(float * output3DLut, int edgeLength,
                                           float * output1DLut, int lutSize, int lutScaleFactor,
                                           ImageColorSpace colorSpace, ImageGammaCurve gammaCurve, ToneMap outputToneMap, RollOff highlightRollOff, unsigned int hdrPeakNits);

}
#endif

