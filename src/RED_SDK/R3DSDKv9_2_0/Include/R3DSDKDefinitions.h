/* R3D SDK library version 9.2 header file. Do *NOT* use this
   header file with any other version of the R3D SDK library!
   
   This header file and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All
   rights reserved. Redistribution of this header is prohibited!
*/

#ifndef R3DSDKDEFINITIONS_H
#define R3DSDKDEFINITIONS_H

#include <stdint.h>

namespace R3DSDK {
    
enum InitializeStatus
{
	ISInitializeOK = 0,
	ISLibraryNotLoaded = 1,
	
	ISR3DSDKLibraryNotFound = 2,
	ISRedCudaLibraryNotFound = 3,
	ISRedOpenCLLibraryNotFound = 4,
	ISR3DDecoderLibraryNotFound = 5,
	ISRedMetalLibraryNotFound = 17,
	
	ISLibraryVersionMismatch = 6,
	
	ISInvalidR3DSDKLibrary = 7,
	ISInvalidRedCudaLibrary = 8,
	ISInvalidRedOpenCLLibrary = 9,
	ISInvalidR3DDecoderLibrary = 10,
	ISInvalidRedMetalLibrary = 18,
	
	ISRedCudaLibraryInitializeFailed = 11,
	ISRedOpenCLLibraryInitializeFailed = 12,
	ISR3DDecoderLibraryInitializeFailed = 13,
	ISR3DSDKLibraryInitializeFailed = 14,
	ISRedMetalLibraryInitializeFailed = 19,
	
	ISInvalidPath = 15,
	ISInternalError = 16,

	ISMetalNotAvailable = 20,
	ISCudaNotAvailable = 21
};
    
// Clip load status
enum LoadStatus
{
	LSClipLoaded = 0,
	LSPathNotFound = 1,					// could not find the clip
	LSFailedToOpenFile = 2,				// could not open the clip
	LSNotAnR3DFile = 3,					// clip does not appear to be a(n) (valid) R3D file
	LSClipIsEmpty = 4,					// clip doesn't have any video frames in it
	LSOutOfMemory = 5,					// no more memory could be allocated
	LSUnknownError = 6,					// unknown error (shouldn't happen)
	LSNoClipOpen = 7,					// initial status, no clip has been loaded yet
	LSNotInitialized = 8				// library wasn't loaded properly (if obj-c, try new() instead)
};

// Clip create status
enum CreateStatus
{
	CSStarted = 0,						// trim has started but has not finished yet
	CSFrameAdded = 1,					// a frame has been added to the output clip, but not finished yet
	CSDone = 2,							// trim is done

	CSOutOfMemory = 3,					// no more memory could be allocated
	CSRequestOutOfRange = 4,			// the requested start or end frame does not exist in the clip
	CSInvalidParameter = 5,
	CSFailedToGetSourceFrame = 6,		// unable to load a frame from the source clip
	CSFailedToCreateDestination = 7,	// unable to create output clip
	CSFailedToWriteToDestination = 8,	// unable to write to output clip
	CSUnknownError = 9,
	CSInvalidSourceClip = 10,			// source clip cannot be used for trim (RED ONE firmware build 15 and below not supported)
	CSInvalidPath = 11,					// output path is invalid (see trim requirements)
	CSFailedToGetSourceAudio = 12,		// unable to load needed audio samples form source clip
	CSOutOfOrder = 13,					// streaming packets need to be added in order
	CSInvalidStream = 14				// streaming packet is invalid
};

// Video decode status
enum DecodeStatus
{
	DSDecodeOK = 0,
	DSOutputBufferInvalid = 1,			// null pointer, too small or not aligned
	DSRequestOutOfRange = 3,			// the requested frame does not exist in the clip
	DSInvalidParameter = 4,				// one of the supplied parameters is not valid
	DSIsDroppedFrame = 5,				// the requested frame is a dropped frame and can't be decoded (video decode only)
	DSDecodeFailed = 6,					// decode failed because of a corrupt frame
	DSOutOfMemory = 7,					// no more memory could be allocated
	DSUnknownError = 8,					// unknown error (shouldn't happen)
	DSNoClipOpen = 9,					// open a clip first before trying to decode a frame
	DSCannotReadFromFile = 10,			// I/O error reading the frame
	DSInvalidPixelType = 11,			// cannot decode to specified pixel type
	DSNotAnHDRxClip = 12,				// cannot decode HDRx as this is not an HDRx clip
	DSCancelled = 13,					// user requested decode to be cancelled
	DSUnsupportedClipFormat = 14,		// this clip format is not supported for the requested decode
	DSParameterUnsupported = 15,		// one of the parameters supplied is not supported by this API. Upgrade to newer dynamic libraries.
	DSDecoderNotOpened = 16				// open decoder first before submitting decodes
};

// Possible marker types
enum MarkerType
{
	MarkerSubClip    = 0,				// region, most likely has different start & end frame numbers
	MarkerStillFrame = 1				// single frame marker, start & end frame number will be the same
};

// Marker type and frame range
struct Marker
{
private:
	unsigned int reserved1;
	unsigned int reserved2;

public:
	MarkerType	Type;
	size_t		StartFrameNumber;
	size_t		EndFrameNumber;

	Marker();
};

enum ImageGammaCurve
{
	ImageGammaLinear = 0,
	ImageGammaSRGB = 8,

	ImageGammaHDR2084 = 14,
	ImageGammaBT1886 = 15,
	ImageGammaLog3G12 = 16,				// mid grey point maps to 1/3, encodes 12 stops above mid grey
	ImageGammaLog3G10 = 18,				// mid grey point maps to 1/3, encodes 10 stops above mid grey
	ImageGammaREDlogFilm = 9,			// only available with color version 2 or newer, otherwise ignored (REDlog will be used)
	
	// Only for IPP2 (ColorVersion3)
	ImageGammaHybridLogGamma = 19,		// color version 3+
	ImageGamma2_2 = 20,					// color version 3+
	ImageGamma2_6 = 21,					// color version 3+

	// Only for Legacy (ColorVersion2), deprecated and not available in IPP2 decode mode.
	ImageGammaRec709 = 1,				// BT1886 will be used in IPP2 mode.
	ImageGammaREDgamma4 = 12,			// v4: only available with color version 2, otherwise ignored

	// Below gamma curves are deprecated and not available in IPP2 decode mode, only for ColorVersion2.
	ImageGammaPDlog685 = 4,
	ImageGammaPDlog985 = 5,
	ImageGammaCustomPDlog = 6,			// if used CustomPDLogBlackPoint, WhitePoint & Gamma must be set!

	ImageGammaREDspace = 2,				// only available with color version 2, otherwise ignored
	ImageGammaREDlog = 3,				// only available with color version 2, otherwise ignored
	ImageGammaREDgamma = 7,				// v1: only available with color version 2, otherwise ignored
	ImageGammaREDgamma2 = 10,			// v2: only available with color version 2, otherwise ignored
	ImageGammaREDgamma3 = 11			// v3: only available with color version 2, otherwise ignored
};

enum ImageColorSpace
{
	ImageColorREDWideGamutRGB = 15,		// only available with color version 2 or newer, otherwise ignored (CameraRGB will be used)

	ImageColorRec2020 = 13,				// only available with color version 2 or newer, otherwise ignored (REDspace will be used)
	ImageColorRec709 = 1,
	ImageColorSRGB = 4,
	ImageColorAdobe1998 = 5,
	ImageColorDCIP3 = 16,				// only available with color version 2 or newer
	ImageColorProPhotoRGB = 17,			// only available with color version 2 or newer
	
	// Only for IPP2 (ColorVersion3)
	ImageColorDCIP3D65 = 18,			// only available with color version 3 or newer

	// Only for Legacy (ColorVersion2), deprecated and not available in IPP2 decode mode.
	ImageColorDRAGONcolor2 = 12,		// v2: only available with color version 2, otherwise ignored
	ImageColorREDcolor4 = 11,			// v4: only available with color version 2, otherwise ignored

	// Below color spaces are deprecated and not available in IPP2 decode mode, only for ColorVersion2.
	ImageColorCameraRGB = 0,
	ImageColorREDspace = 2,
	ImageColorREDcolor = 3,				// v1: only available with color version 2, otherwise ignored
	ImageColorREDcolor2 = 6,			// v2: only available with color version 2, otherwise ignored
	ImageColorREDcolor3 = 8,			// v3: only available with color version 2, otherwise ignored
	ImageColorDRAGONcolor = 9			// v1: only available with color version 2, otherwise ignored
};

// IPP2 (ColorVersion3) only
enum ImagePipeline
{
	Primary_Development_Only = 0,		// output in REDWideGamutRGB & Log3G10. Only Kelvin, Tint, ISO and ExposureAdjust available.
	Full_Graded = 1						// default
};

// IPP2 (ColorVersion3) only
enum RollOff
{
	RollOff_Very_Soft = 4,
	RollOff_Soft = 3,
	RollOff_Medium = 2,
	RollOff_Hard = 1,
    RollOff_None = 0					// advanced option
};

// IPP2 (ColorVersion3) only
enum ToneMap
{
	ToneMap_Low = 0,
	ToneMap_Medium = 1,
	ToneMap_High = 2,
	ToneMap_None = 3
};

// This structure gives access to the default, minimum and maximum setting
// for each of the image processing parameters. In a few cases it also
// provides a list of the possible choices. You are encouraged to use this
// data in your User Interface so new options in future versions are
// automatically added without you having to do anything.
struct ImageProcessingLimits
{
	static const float              KelvinMin, KelvinMax, KelvinDefault;
	static const float              TintMin, TintMax, TintDefault;
	static const float              ExposureMin, ExposureMax, ExposureDefault;
	static const float              GainsMin, GainsMax, GainsDefault;
	static const float              SaturationMin, SaturationMax, SaturationDefault;
	static const float              ContrastMin, ContrastMax, ContrastDefault;
	static const float              BrightnessMin, BrightnessMax, BrightnessDefault;
	static const float              DRXMin, DRXMax, DRXDefault;
	static const float              ShadowMin, ShadowMax, ShadowDefault;
	static const float              FLUTMin, FLUTMax, FLUTDefault;

	static const float              LggLiftMin , LggLiftMax , LggLiftDefault;
	static const float              LggGammaMin, LggGammaMax, LggGammaDefault;
	static const float              LggGainMin , LggGainMax , LggGainDefault;

	static const size_t				ISOCount;			// Number of ISO values in ISOList below
	static const size_t				ISOList[];			// List of valid ISO values	
	static const size_t				ISODefault;			// default ISO value

	static const size_t				GammaCurveCount;	// Number of gamma curve values in GammaCurveList below
	static const char *				GammaCurveLabels[];	// ImageGammaCustomPDlog is not included because it's not a preset
	static const ImageGammaCurve	GammaCurveMap[];	// List of gamma options
	static const ImageGammaCurve	GammaCurveDefault;	// default gamma value

	static const size_t				ColorSpaceCount;	// Number of color space values in ColorSpaceList below
	static const char *				ColorSpaceLabels[];
	static const ImageColorSpace	ColorSpaceMap[];	// List of color space options
	static const ImageColorSpace	ColorSpaceDefault;	// default color space value

	static const size_t             CustomPDLogBlackPointMin, CustomPDLogBlackPointMax, CustomPDLogBlackPointDefault;
	static const size_t             CustomPDLogWhitePointMin, CustomPDLogWhitePointMax, CustomPDLogWhitePointDefault;
	static const float              CustomPDLogGammaMin, CustomPDLogGammaMax, CustomPDLogGammaDefault;

	static const size_t				ImagePipelineModeCount;
	static const char *				ImagePipelineModeLabels[];
	static const ImagePipeline		ImagePipelineModeMap[];
	static const ImagePipeline		ImagePipelineModeDefault;

	static const size_t				HighlightRollOffCount;
	static const char *				HighlightRollOffLabels[];
	static const RollOff			HighlightRollOffMap[];
	static const RollOff			HighlightRollOffDefault;

	static const size_t				OutputToneMapCount;
	static const char *				OutputToneMapLabels[];
	static const ToneMap			OutputToneMapMap[];
	static const ToneMap			OutputToneMapDefault;

	static const float              ExposureAdjustMin, ExposureAdjustMax, ExposureAdjustDefault;

	static const unsigned int       HdrPeakNitsMin, HdrPeakNitsMax, HdrPeakNitsDefault;

	static const float              CdlSlopeMin, CdlSlopeMax, CdlSlopeDefault;
	static const float              CdlOffsetMin, CdlOffsetMax, CdlOffsetDefault;
	static const float              CdlPowerMin, CdlPowerMax, CdlPowerDefault;

private:
	ImageProcessingLimits();							// all data is static, don't instantiate this struct
};

// Amount of detail extraction (not sharpening!) for
// the full resolution premium software decode only!
enum ImageDetail
{
	ImageDetailLow = 0,
	ImageDetailMedium = 1,
	ImageDetailHigh = 2									// default
};

// Compensation for the OLPF (low-pass) filter for
// the half & full resolution premium software decodes
enum ImageOLPFCompensation
{
	ImageOLPFCompOff = 0,
	ImageOLPFCompLow = 1,
	ImageOLPFCompMedium = 2,
	ImageOLPFCompHigh = 3
};

// Amount of noise reduction to do for the
// full resolution premium software decode
enum ImageDenoise
{
	ImageDenoiseOff = 0,
	ImageDenoiseMinimum = 1,
	ImageDenoiseMilder = 2,
	ImageDenoiseMild = 3,
	ImageDenoiseMedium = 4,
	ImageDenoiseStrong = 5,
	ImageDenoiseMaximum = 6
};

enum FlashingPixelAdjust
{
	FlashingPixelAdjustOff = 0,
	FlashingPixelAdjustMild = 0x46504D44,
	FlashingPixelAdjustMedium = 0x46504D4D,
	FlashingPixelAdjustStrong = 0x46505347
};

// Which color version to use with the image processing settings.
// Default is ColorVersion3, which is the new IPP2 color science.
enum ColorVersion
{
	ColorVersion1  = 1,		// Legacy
	ColorVersion2  = 2,		// Legacy (FLUT)
	ColorVersion3  = 3,		// IPP2
	ColorVersionBC = 4		// Broadcast, only usable on Broadcast clips
};

// Legacy (FLUT) only. Lift, Gamma and Gain settings.
struct LiftGammaGain
{
	float	Lift;										// -1 -- 1
	float	Gamma;										//  0 -- 4
	float	Gain;										//  0 -- 2
};

// IPP2 only, Color Decision List settings.
struct SlopeOffsetPower
{
	float	Slope;
	float	Offset;
	float	Power;
};

// IPP2 only
typedef void * Handle3DLut;

// IPP2 only
enum LensCorrection
{
	LensCorrectionAuto = 0,	// follow metadata (user has not overridden the setting)
	LensCorrectionOn   = 1,	// force correction on, if correction metadata is present
	LensCorrectionOff  = 2	// force correction off
};

// Settings to process the image with. If these settings are not
// supplied the decoder will use the clip's default settings.
struct ImageProcessingSettings
{
private:
	unsigned int reserved1;
	unsigned int reserved2;
	bool reserved3;
	bool reserved4;

public:
	// *********************************************************************
	// Color version defaults to the latest and greatest (V3 for the current
	// SDK release). With the exception of FLUT and Shadow, all image
	// processing setings are available for both color versions. For older
	// clips you may want to offer a backwards compatibility decode setting.
	// To do this set Version to ColorVersion1. FLUT, Shadow, as well as
	// REDgamma and REDcolor options will not be available with the old color!
	// Settings out of bounds (see ImageProcessingLimits) will be clipped.
	// *********************************************************************
	ColorVersion			Version;					// if ColorVersion1 is set for a ColorVersion2 clip it will be forced to ColorVersion2 etc.

	float					Kelvin;						// Kelvin color temperature for white balance
	float					Tint;						// Tint for white balance
	
	float					ExposureCompensation;		// Converted in ColorVersion3 mode.
	float					GainRed, GainGreen, GainBlue;// Ignored with ColorVersion3 or ColorVersionBC
	float					Saturation;					// Ignored with ColorVersion3 or ColorVersionBC
	float					Contrast;					// Ignored with ColorVersion3 or ColorVersionBC
	float					Brightness;					// Converted in ColorVersion3 mode. Ignored with ColorVersionBC
	float					DRX;						// Highlight extension control. Ignored for Dragon or Helium clips or in ColorVersion3 & ColorVersionBC modes.

	// See ImageProcessingLimits::ISOList for the valid list of ISO values
	size_t					ISO;						// will clamp to the nearest valid setting
	ImageGammaCurve			GammaCurve;					// limited to ImageGammaBT1886 or ImageGammaHDR2084 with ColorVersionBC
	ImageColorSpace			ColorSpace;					// limited to ImageColorRec709 or ImageColorRec2020 with ColorVersionBC

	// Set all of these fields when GammaCurve is set to ImageGammaCustomPDlog.
	// Ignored with ColorVersion3 (IPP2) or ColorVersionBC.
	size_t					CustomPDLogBlackPoint;
	size_t					CustomPDLogWhitePoint;
	float					CustomPDLogGamma;

	// User defined 5-point luma spline curve. Each of the 5 points consists of
	// an X & Y coordinate (0 -- 1) from lower left to upper right. Array
	// goes: black X, black Y, low X, low Y, mid X, mid Y, high X, high Y, white X, white Y
	// The curve coordinate values should be constrained in both UI and
	// generating code so that the x coordinates always lie such that:
	// black X < low X < mid X < high X < white X. If this is not the case
	// the curve will default to a straight line (0.0, 0.0, 0.25, 0.25,
	// 0.50, 0.50, 0.75, 0.75, 1.0, 1.0). Ignored with ColorVersionBC.
	float					UserCurve[10];				// Luma user curve.

	// Following setting is only used for full res premium decode.
	// THIS SETTING IS IGNORED IN COLORVERSION3 & COLORVERSIONBC!
	ImageDetail				Detail;
	
	// Following is only used for full & half res premium decode.
	// THIS SETTING IS IGNORED IN COLORVERSION3 & COLORVERSIONBC!
	ImageOLPFCompensation	OLPFCompensation;

	// Following setting is only used for full res premium decode.
	// THIS SETTING IS IGNORED IN COLORVERSION3 & COLORVERSIONBC!
	ImageDenoise			Denoise;

	// Following setting is only used for full & half res premium decode.
	union
	{
		// Dragon Enhanced Blacks -- Legacy pipe and Dragon or newer sensors only.
		bool				DEB;

		// IPP2 & Broadcast only
		bool				ChromaNoiseReduction;
	};

	// Following setting is only used for full & half res premium decode.
	// Ignored for N-RAW clips.
	FlashingPixelAdjust		FlashingPixelAdjustment;

	// *********************************************************************
	// Following settings are only available for ColorVersion2 (Legacy)
	// Not available in IPP2 decode mode
	// *********************************************************************

	float					Shadow;						// Ignored with ColorVersion3 & ColorVersionBC.
	float					FLUT;						// Fine ISO control. Ignored with ColorVersion3 & ColorVersionBC.

	// Red, Green & Blue user defined spline curves. Works the same as the luma UserCurve. Ignored with ColorVersionBC.
	float					UserCurveRed[10];
	float					UserCurveGreen[10];
	float					UserCurveBlue[10];

	// Post lift, gamma & gain for Red, Green and Blue channels.
	// Ignored with ColorVersion3 & ColorVersionBC.
	LiftGammaGain			LggRed;
	LiftGammaGain			LggGreen;
	LiftGammaGain			LggBlue;

	// *********************************************************************
	// Following settings are fully available for ColorVersion3 (IPP2) only
	// Some are available for ColorVersionBC (Broadcast) as well
	// *********************************************************************

	ImagePipeline			ImagePipelineMode;

	float					ExposureAdjust;				// also available with ColorVersionBC

	float					CdlSaturation;
	SlopeOffsetPower		CdlRed;
	SlopeOffsetPower		CdlGreen;
	SlopeOffsetPower		CdlBlue;
	bool					CdlEnabled;

	Handle3DLut				Lut3D;						// NULL if disabled/not set. Value is unique per process and cannot
														// be saved and then re-used later. Use CreateOrUpdateRmd() for that.
	bool					Lut3DEnabled;				// defaults to FALSE. Only used if Lut3D is set.

	ToneMap					OutputToneMap;
	RollOff					HighlightRollOff;			// no effect when a LOG curve is applied
	unsigned int			HdrPeakNits;				// HDR2084 only, and HighlightRollOf != RollOff_None

	bool					MasklineAdjust;				// also available with ColorVersionBC, ignored for N-RAW clips

	// Following setting is only used for full & half res premium decode.
	LensCorrection			LensDistortionCorrection;	// recommend hiding or disabling UI if RMD_LENS_DISTORTION_CORRECTION metadata is 0

	// *********************************************************************
	// Function definitions
	// *********************************************************************

	// Sets all fields to their defaults. Keep in mind that you should
	// load the actual clip settings from metadata before setting up
	// the User Interface or using these settings to decode a frame!
	ImageProcessingSettings();

	// Check to see if all parameters are within the proper range and clip any that aren't
	void CheckBounds();
};

// The different resolutions and qualities the clip can be
// decoded at. This list expand over time.
enum VideoDecodeMode
{
	DECODE_FULL_RES_PREMIUM		= 0x44465250,			// 'DFRP', full resolution, slow, but highest resolution & quality
	
	DECODE_HALF_RES_PREMIUM		= 0x44485250,			// 'DHRP', half resolution, slow, but highest quality
	DECODE_HALF_RES_GOOD		= 0x44485247,			// 'DHRG', half resolution, fast, still very good quality
	
	DECODE_QUARTER_RES_GOOD		= 0x44515247,			// 'DQRG', quarter resolution, fast, very good quality
	DECODE_EIGHT_RES_GOOD		= 0x44455247,			// 'DERG', eight resolution, fast, good quality
	DECODE_SIXTEENTH_RES_GOOD	= 0x44535247			// 'DSRG', sixteenth resolution, fast, good quality
};

// The different pixel types to decode images at. All are
// supported for CPU decoding, but not all are for GPU.
enum VideoPixelType
{
	// Supported by software and GPU decodes
	PixelType_16Bit_RGB_Interleaved		= 0x52423649,	// Interleaved RGB decoding in 16-bits per pixel

	PixelType_HalfFloat_RGB_Interleaved = 0x52424846,	// 16-bit half-float decoding. These ImageProcessingSettings fields are ignored:
														//		- GammaCurve (will always be linear)
														//		- Contrast
														//		- Brightness
														//		- Saturation
														//		- RGB gains
														//		- Shadow
														//		- UserCurve, UserCurveRed, UserCurveGreen & UserCurveBlue
														//		- LggRed, LggGreen & LggBlue
														//		- CdlSaturation, CdlRed, CdlGreen, CdlBlue & CdlEnabled
														//		- CustomPDLogBlackPoint, CustomPDLogWhitePoint & CustomPDLogGamma

	PixelType_HalfFloat_RGB_ACES_Int	= 0x52424841,	// 16-bit half-float ACES decoding. These ImageProcessingSettings fields are ignored:
														//		- Same fields as PixelType_HalfFloat_RGB_Interleaved pixel type
														//		- ColorSpace (will always be set to ACES AP0)
	
	// Supported by CPU decode only
	PixelType_16Bit_RGB_Planar			= 0x52423650,	// Planar RGB decoding in 16-bits per pixel

	PixelType_8Bit_BGR_Interleaved		= 0x42475238,	// Interleaved BGR 8-bit
	PixelType_8Bit_BGRA_Interleaved		= 0x42524138,	// Interleaved BGRA decoding in 8 bits per pixel, alpha channel = 0xFF

	PixelType_10Bit_DPX_MethodB			= 0x44503042	// Interleaved RGB 10-bit DPX Method B
};

// HDRx blending algorithm to use when doing HDR blending
enum HdrBlendAlgorithm
{
	HDRx_SIMPLE_BLEND					= 0x48445253,	// Simple blend, exactly as implemented in 12_SimpleHDRxBlend
	HDRx_MAGIC_MOTION					= 0x4844524D	// Magic Motion, more sophisticated blend
};

// This structure gives access to the default, minimum and maximum setting
// for each of the HDR processing parameters. In a few cases it also
// provides a list of the possible choices. You are encouraged to use this
// data in your User Interface so new options in future versions are
// automatically added without you having to do anything.
struct HdrProcessingLimits
{
	static const size_t					BlendAlgorithmCount;
	static const char *					BlendAlgorithmLabels[];
	static const HdrBlendAlgorithm		BlendAlgorithmMap[];
	static const HdrBlendAlgorithm		BlendAlgorithmDefault;

	static const float					BiasMin, BiasMax, BiasDefault;

private:
	HdrProcessingLimits();								// all data is static, don't instantiate this struct
};

// HDR blending settings. This structure needs to be constructed
// to submit a HDR blending job to the decoder. Every field must
// be set.
struct HdrProcessingSettings
{
private:
	unsigned int reserved1;
	unsigned int reserved2;

public:
	// HDRx blending algorithm to use
	HdrBlendAlgorithm			BlendAlgorithm;

	// Amount of blending for the modes, must be in range of -1 -- +1 (will be clipped if it's not!)
	float						Bias;					// Bias is mode dependent. For Simple Blend it controls the blend
														// between the two images, +1 being full normal exposure, -1 being
														// full highlight track exposure and 0 being an equal blend of the two.
														// For Magic Motion, it adjusts the amount of highlight rolloff in the
														// blended HDR image. +1 is no highlight rolloff and -1 is full rolloff

	// Sets all fields to their defaults.
	HdrProcessingSettings();

	// Check to see if all parameters are within the proper range and clip any that aren't
	void CheckBounds();
};

// HDR read status & write setting when reading & writing RMD sidecar files
enum HdrMode
{
	HDR_USE_TRACKNO,									// Use A / main (#0) or X (#1) track
	HDR_DO_BLEND										// Blend A and X tracks using specified settings
};

class Metadata;

// The decode job structure. This structure needs to be constructed
// to submit a decode job to the decoder. Every field must be set,
// with the exception of 'ImageProcessing' & "HdrProcessing'
struct VideoDecodeJob
{
private:
	unsigned int reserved1;
	unsigned int reserved2;
	void * reserved3;

public:
	// Resolution/speed to decode the image at. This will also 
	// influence how much memory is needed for the output buffer
	VideoDecodeMode				Mode;

	// Which pixel type to decode the image in. See the VideoPixelType
	// enum for the available options. To get the image in the
	// original pixel type supported by earlier SDK's set
	// PixelType_16Bit_RGB_Planar
	VideoPixelType				PixelType;

	// Pointer to the buffer to store the image in. This cannot be
	// NULL otherwise the decode will fail. The buffer must be aligned
	// on a 16-byte boundary (see sample code).
	void *						OutputBuffer;

	// Size ouf the output buffer. If this value is too small
	// the decode will fail.
	size_t						OutputBufferSize;
	
	// Image processing settings to apply to the decode. This
	// parameter is optional (set it to NULL if not used!).
	// The decoder will simply use the clip's default settings
	// if no image processing structure is used to override.
	ImageProcessingSettings	*	ImageProcessing;

	// HDRx processing settings to apply to the decode. This
	// parameter is optional (set it to NULL if not used!).
	// If the clip is not an HDRx clip, the decode will fail.
	// If the clip is an HDRx clip and no settings are supplied
	// the main track (A track) will be decoded, or the specified
	// track when using VideoTrackDecodeFrame(). Video track
	// number is ignored when this field is not NULL and you're
	// using VideoTrackDecodeFrame() API!
	HdrProcessingSettings *		HdrProcessing;
	
	// Optionally return the decoded frame's per-frame metadata
	Metadata *					OutputFrameMetadata;

	VideoDecodeJob();
};

struct IMUSample
{
	struct Axes
	{
		float x, y, z;
	};

	uint64_t timestamp;		// in microseconds, see RMD_FRAME_TIMESTAMP + Clip::GetFrameMetadata()
	Axes acceleration;		// in G's
	Axes rotation;			// in deg/s
};

}
#endif

