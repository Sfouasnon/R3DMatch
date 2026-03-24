/* R3D SDK library version 9.2 header file. Do *NOT* use this
   header file with any other version of the R3D SDK library!
   
   This header file and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All
   rights reserved. Redistribution of this header is prohibited!
*/

#ifndef R3DSDKMETADATA_H
#define R3DSDKMETADATA_H

namespace R3DSDK {

// Types of data that can be stored in the metadata database
enum MetadataType
{
	MetadataTypeInvalid = 0,
	MetadataTypeInt     = 1,
	MetadataTypeString  = 2,
	MetadataTypeFloat   = 3
};

// For all metadata items an int, float & string output can be requested.
// Below a table showing the conversions that can happen:
//
// Original type | Requested type | Description
// --------------|----------------|-----------------------------------------
// Int           | Float          | Int to float
// Int           | String         | Int to string, no padding or anything
// --------------|----------------|-----------------------------------------
// Float         | Int            | Rounded float to int (23.976 becomes 24)
// Float         | String         | Float to string, no padding or anything
// --------------|----------------|-----------------------------------------
// String        | Int            | Length of string
// String        | Float          | Length of string converted to float
//
// The MetadataItemType() methods return the original data type so you
// can avoid any conversions if that's desirable.
//
// Most methods have a key as well as index variant. The former is most useful
// if you exactly know which metadata item to query, the search is case
// insensitive. The latter is most useful when enumerating through the metadata
// list. When requesting items by key make sure to call MetadataExists() to
// be sure the item actually exists in the database. Otherwise you will get
// a return value of 0, 0.0 or "" depending on which output type you request.
//
// Below a list of items that might exist in the metadata database. These keys
// can be used to quickly get a specific metadata item. You CANNOT depend on
// the metadata items existing at a specific index!! Keep in mind that only a
// couple of items are guaranteed to exist in the database.
// DO NOT COUNT ON ALL ITEMS BEING AVAILABLE
static const char RMD_3D_LUT[] =						"3d_lut";						// IPP2 only, optional filename of 3D LUT as set in camera
static const char RMD_3D_LUT_DEFAULT_ENABLED[] =		"3d_lut_default_enabled";		// IPP2 only, whether 3D LUT, if not overridden, should be enabled for this clip
static const char RMD_ANAMORPHIC_MODE[] =				"anamorphic_mode";				// different form of pixel aspect ratio. 0 = not anamorphic, 1 = 2x, 2 = 1.3x, 3 = 1.25x, 4 = 1.6x, 5 = 1.8x, 6 = 1.5x
static const char RMD_BRIGHTNESS[] =					"brightness";					// Also available through the image processing settings structure
static const char RMD_BROADCAST_SETTINGS[] =			"broadcast_settings";			// Broadcast clip record & develop settings when decoded with ColorVersionBC. Has embedded '\n' & padding for display.
static const char RMD_CAMERA_ACCELERATION_X[] =			"camera_acceleration_x";		// left/right in g
static const char RMD_CAMERA_ACCELERATION_Y[] =			"camera_acceleration_y";		// forward/back in g
static const char RMD_CAMERA_ACCELERATION_Z[] =			"camera_acceleration_z";		// up/down in g
static const char RMD_CAMERA_FIRMWARE_REVISION[] =		"camera_firmware_revision";
static const char RMD_CAMERA_FIRMWARE_VERSION[] =		"camera_firmware_version";
static const char RMD_CAMERA_ID[] =						"camera_id";					// Camera ID as set in camera
static const char RMD_CAMERA_MODEL[] =					"camera_model";					// "RED ONE", "EPIC-M", "EPIC-X", "SCARLET-X", "WEAPON", "RAVEN" etc.
static const char RMD_CAMERA_MODEL_ID[] =				"camera_model_id";				// Unique ID for each camera model
static const char RMD_CAMERA_NETWORK_NAME[] =			"camera_network_name";
static const char RMD_CAMERA_PIN[] =					"camera_pin";					// Camera's PIN (serial number)
static const char RMD_CAMERA_POSITION[] =				"camera_position";				// Optional. Usually 'C'. 3D uses 'L' or 'R'. Can be any character in 'A' - 'Z' range
static const char RMD_CAMERA_ROTATION_X[] =				"camera_rotation_x";			// tilt in deg/s
static const char RMD_CAMERA_ROTATION_Y[] =				"camera_rotation_y";			// pan in deg/s
static const char RMD_CAMERA_ROTATION_Z[] =				"camera_rotation_z";			// roll in deg/s
static const char RMD_CDL_DEFAULT_ENABLED[] =			"cdl_default_enabled";			// IPP2 only, whether CDL control, if not overridden, should be enabled for this clip
static const char RMD_CDL_FILENAME[] =					"cdl_filename";					// IPP2 only, optional filename of ASC CDL side car file as set in camera
static const char RMD_CLIP_ASPECT_RATIO[] =				"clip_aspect_ratio";
static const char RMD_CLIP_ASPECT_DENOMINATOR[] =		"clip_aspect_ratio_denominator";
static const char RMD_CLIP_ASPECT_NUMERATOR[] =			"clip_aspect_ratio_numerator";
static const char RMD_CLIP_DEFAULT_COLOR_VERSION[] =	"clip_default_color_version";	// Clip default ColorVersion. Will be ColorVersion2 (Legacy), or ColorVersion3 (IPP2) if set in camera.
static const char RMD_CLIP_ID[] =						"clip_id";						// Clip ID as sset in camera (001 - 999)
static const char RMD_CLIP_UUID[] =						"clip_uuid";					// Text representation of Clip::Uuid(). No guarantee this follows the ITU-T Rec. X.667 / ISO/IEC 9834-8:2005 standards.
static const char RMD_CONTRAST[] =						"contrast";						// Also available through the image processing settings structure
static const char RMD_DIGITAL_GAIN_BLUE[] =				"digital_gain_blue";			// Also available through the image processing settings structure
static const char RMD_DIGITAL_GAIN_GREEN[] =			"digital_gain_green";			// Also available through the image processing settings structure
static const char RMD_DIGITAL_GAIN_RED[] =				"digital_gain_red";				// Also available through the image processing settings structure
static const char RMD_DROPPED_FRAME_COUNT[] =			"dropped_frame_count";
static const char RMD_DRX[] =							"drx";							// 0.0 (off) - 1.0 (fully on)
static const char RMD_EXPOSURE_ADJUST[] =				"exposure_adjust";				// IPP2 only.
static const char RMD_EXPOSURE_COMPENSATION[] =			"exposure_compensation";		// Legacy only. Guaranteed to exist, also available through the image processing settings structure
static const char RMD_EXPOSURE_TIME[] =					"exposure_time";				// Length the frame was exposed for in microseconds
static const char RMD_EXTERNAL_TALLY[] =				"external_tally";				// 0 = off, otherwise tally number
static const char RMD_EXTERNAL_TALLY_COLOR[] =			"external_tally_color";			// RGBA
static const char RMD_FLIP_HORIZONTAL[] =				"flip_horizontal";				// 0 if image does not need to be flipped horizontally, 1 if it should
static const char RMD_FLIP_VERTICAL[] =					"flip_vertical";				// 0 if image does not need to be flipped vertically, 1 if it should
static const char RMD_FLUT_CONTROL[] =					"flut_control";					// Legacy.
static const char RMD_FRAME_GUIDE_ENABLED[] =			"frame_guide_enabled";			// 0 = disabled, 1 = enabled
static const char RMD_FRAME_GUIDE_NAME[] =				"frame_guide_name";
static const char RMD_FRAME_GUIDE_TOPLEFT_X[] =			"frame_guide_topleft_x";		// float, 0.0 - 1.0, multiply by image width - 1
static const char RMD_FRAME_GUIDE_TOPLEFT_Y[] =			"frame_guide_topleft_y";		// float, 0.0 - 1.0, multiply by image height - 1
static const char RMD_FRAME_GUIDE_WIDTH[] =				"frame_guide_width";			// float, 0.0 - 1.0, multiply by image width
static const char RMD_FRAME_GUIDE_HEIGHT[] =			"frame_guide_height";			// float, 0.0 - 1.0, multiply by image height
static const char RMD_FRAME_ABSOLUTE_TIMECODE[] =		"frame_absolute_timecode";		// per-frame API only: Absolute/TOD timecode as recorded
static const char RMD_FRAME_EDGE_TIMECODE[] =			"frame_edge_timecode";			// per-frame API only: Edge/Run-record timecode as recorded
static const char RMD_FRAME_PTP_TIMESTAMP[] =			"frame_ptp_timestamp";			// per-frame API only: External PTP timestamp in nanoseconds. 64-bit unsigned int stored as a string
static const char RMD_FRAME_TIMESTAMP[] =				"frame_timestamp";				// per-frame API only: Internal camera timestamp in microseconds. 64-bit unsigned int stored as a string
static const char RMD_FRAMERATE[] =						"framerate";					// Calculated from RMD_FRAMERATE_NUMERATOR / RMD_FRAMERATE_DENOMINATOR
static const char RMD_FRAMERATE_DENOMINATOR[] =			"framerate_denominator";		// Guaranteed to exist (also true for RMD_FRAMERATE above)
static const char RMD_FRAMERATE_NUMERATOR[] =			"framerate_numerator";			// Guaranteed to exist
static const char RMD_GENLOCK_SETTING[] =				"genlock_setting";				// 0 = not genlocked, 1 = genlocked to external source
static const char RMD_GMT_DATE[] =						"gmt_date";
static const char RMD_GMT_TIME[] =						"gmt_time";
static const char RMD_HDR_MODE[] =						"hdr_mode";						// 0 = no HDR, 2 = HDRx, 3 = Extended Highlights
static const char RMD_HDR_STOPS_OVER[] =				"hdr_stops_over";				// number f-stops the X track is over the A (main) track
static const char RMD_IMAGE_HEIGHT[] =					"image_height";					// Guaranteed to exist, also available as clip property
static const char RMD_IMAGE_WIDTH[] =					"image_width";					// Guaranteed to exist, also available as clip property
static const char RMD_ISO[] =							"iso";							// Guaranteed to exist, also available through the image processing settings structure
static const char RMD_ISO_CALIBRATION_VERSION[] =		"iso_calibration_version";		// 1 = V1: original pre-IPP2 system, 2 = V2: IPP2 calibration system
static const char RMD_JAMSYNC_SETTING[] =				"jamsync_setting";				// 0 = internal clock, 1 = external clock used
static const char RMD_LENS_APERTURE[] =					"lens_aperture";				// Aperture * 10, 0xFFFF = fully closed
static const char RMD_LENS_APERTURE_LABEL[] =			"lens_aperture_label";			// f2.8 etc
static const char RMD_LENS_APERTURE_UNITS[] =			"lens_aperture_units";			// 0 = f-stop, 1 = T-stop
static const char RMD_LENS_BRAND[] =					"lens_brand";
static const char RMD_LENS_COOKE_I_STATIC[] =			"lens_cooke_i_static";			// Cooke /i static lens data
static const char RMD_LENS_COOKE_I_DYNAMIC[] =			"lens_cooke_i_dynamic";			// Cooke /i dynamic lens data (available through the per-frame metadata APIs), HEX encoded binary string with '/' separators
static const char RMD_LENS_DISTANCE_UNITS[] =			"lens_distance_units";			// 0 = Metric, 1 = Imperial
static const char RMD_LENS_DISTORTION_CORRECTION[] =	"lens_distortion_correction";	// 0 = not available, 1 = available, default on, 2 = available, but default off
static const char RMD_LENS_FOCAL_LENGTH[] =				"lens_focal_length";			// in mm
static const char RMD_LENS_FOCAL_LENGTH_MICROMETERS[] = "lens_focal_length_micrometers";// current (zoom) focal length in micrometers (set over RCP)
static const char RMD_LENS_FOCUS_DISTANCE[] =			"lens_focus_distance";			// in mm, 0xFFFFFFFF = infinite
static const char RMD_LENS_FOCUS_DISTANCE_NEAR[] =		"lens_focus_distance_near";
static const char RMD_LENS_FOCUS_DISTANCE_FAR[] =		"lens_focus_distance_far";
static const char RMD_LENS_HYPERFOCAL_DISTANCE[] =		"lens_hyperfocal_distance";		// in mm
static const char RMD_LENS_MOUNT[] =					"lens_mount";					// if present will be the name of the mount
static const char RMD_LENS_NAME[] =						"lens_name";
static const char RMD_LENS_OWNER[] =					"lens_owner";
static const char RMD_LENS_SERIAL_NUMBER[] =			"lens_serial_number";
static const char RMD_LINKED_CAMERA_SETUP[] =			"linked_camera_setup";			// 0 = camera wasn't linked, 1 = clip was recorded on master camera, 2 = clip was recorded on (a) slave camera
static const char RMD_LOCAL_DATE[] =					"local_date";
static const char RMD_LOCAL_TIME[] =					"local_time";
static const char RMD_MEDIA_SERIAL_NUMBER[] =			"media_serial_number";
static const char RMD_MONITOR_SHARPNESS[] =				"monitor_sharpness";			// in percent
static const char RMD_MOTION_MOUNT_ND_STOPS[] =			"motion_mount_nd_stops";		// DSMC Motion Mount ND value in stops
static const char RMD_MOTION_MOUNT_SHUTTER_TYPE[] =		"motion_mount_shutter_type";	// DSMC Motion Mount Shutter Type: 0 = ND only, 1 = Soft Shutter, 2 = Square Shutter
static const char RMD_ND_STOPS[] =						"nd_stops";						// Built-in or Motion Mount ND value
static const char RMD_ORIGINAL_FILENAME[] =				"original_filename";			// original camera filename for first R3D file in the clip
static const char RMD_OUTPUT_TRANSFORM[] =				"output_transform";				// 0 = should override to preffered color & gamma, 1 = IPP2 output should be decoded to REDWideGamutRGB & Log3G10 only
static const char RMD_PHANTOM_TRACK_COUNT[] =			"phantom_track_count";			// 0 = not phantom track clip, otherwise count of clips
static const char RMD_PHANTOM_TRACK_FRAME_GAP[] =		"phantom_track_frame_gap_ns";	// 0 = not phantom track clip, otherwise frame gap in nanoseconds
static const char RMD_PHANTOM_TRACK_INDEX[] =			"phantom_track_index";			// 0 = not phantom track clip, 1 = first clip, 2 = second clip
static const char RMD_PIXEL_ASPECT_RATIO[] =			"pixel_aspect_ratio";			// 1.0 normally, otherwise anamorphic desqueeze factor
static const char RMD_RECORD_FRAMERATE[] =				"record_framerate";				// Framerate the recording was made at = RMD_RECORD_FRAMERATE_NUMERATOR / RMD_RECORD_FRAMERATE_DENOMINATOR
static const char RMD_RECORD_FRAMERATE_DENOMINATOR[] =	"record_framerate_denominator";	// optional
static const char RMD_RECORD_FRAMERATE_NUMERATOR[] =	"record_framerate_numerator";	// optional
static const char RMD_REDCODE[] =						"redcode";						// String describing the REDCODE format the clip was recorded with
static const char RMD_REEL_ID[] =						"reel_id";						// Reel ID as set in camera
static const char RMD_REEL_ID_FULL[] =					"reel_id_full";					// REDCINE-X PRO style Reel ID Full
static const char RMD_REEL_ID_8_CHAR[] =				"reel_id_8_character";			// REDCINE-X PRO style Reel ID 8 Character for EDL's
static const char RMD_RESOLUTION_FORMAT_NAME[] =		"resolution_format_name";		// Name for the clip format (resolution)
static const char RMD_SATURATION[] =					"saturation";					// Also available through the image processing settings structure
static const char RMD_SENSOR_ID[] =						"sensor_id";					// Unique ID for each sensor model
static const char RMD_SENSOR_NAME[] =					"sensor_name";					// "MYSTERIUM 4.5K S35", "MYSTERIUM-X 5K S35", "DRAGON 6K S35", "HELIUM 8K S35", etc.
static const char RMD_SENSOR_OLPF_INTERCHANGEABLE[] =	"sensor_olpf_interchangeable";	// 0 = no, 1 = yes
static const char RMD_SENSOR_OLPF_NAME[] =				"sensor_olpf_name";				// if camera has interchangeable sensor OLPF, this will contain the name of the OLPF installed
static const char RMD_SENSOR_SENSITIVITY_ID[] =			"sensor_sensitivity_id";		// 0 = standard, 1 = low-light
static const char RMD_SENSOR_SENSITIVITY_NAME[] =		"sensor_sensitivity_name";
static const char RMD_SHADOW[] =						"shadow";
static const char RMD_SHUTTER_DEGREES[] =				"shutter_degrees";				// Shutter in degrees, calculated from RMD_EXPOSURE_TIME & RMD_FRAMERATE
static const char RMD_SHUTTER_FRACTIONS[] =				"shutter_fractions";			// Shutter in fractions (1/RMD_SHUTTER_FRACTIONS), see notes for shutter degrees
static const char RMD_SHUTTER_PHASE_OFFSET[] =			"shutter_phase_offset";
static const char RMD_START_ABSOLUTE_TIMECODE[] =		"start_absolute_timecode";		// Guaranteed to exist, in the form of "01:00:00:00"
static const char RMD_START_EDGE_TIMECODE[] =			"start_edge_timecode";			// Guaranteed to exist, in the form of "01:00:00:00"
static const char RMD_STEREO_SETUP[] =					"stereo_setup";					// 0 = not a stereo clip, 1 = left eye, 2 = right eye
static const char RMD_USER_ACTION_GUIDE_ENABLED[] =		"user_action_guide_enabled";	// 0 = disabled, 1 = enabled
static const char RMD_USER_ACTION_GUIDE_TOPLEFT_X[] =	"user_action_guide_topleft_x";	// float, 0.0 - 1.0, multiply by image width - 1
static const char RMD_USER_ACTION_GUIDE_TOPLEFT_Y[] =	"user_action_guide_topleft_y";	// float, 0.0 - 1.0, multiply by image height - 1
static const char RMD_USER_ACTION_GUIDE_WIDTH[] =		"user_action_guide_width";		// float, 0.0 - 1.0, multiply by image width
static const char RMD_USER_ACTION_GUIDE_HEIGHT[] =		"user_action_guide_height";		// float, 0.0 - 1.0, multiply by image height
static const char RMD_USER_CURVE_BLACK_X[] =			"user_curve_black_x";			// Also available through the image processing settings structure
static const char RMD_USER_CURVE_BLACK_Y[] =			"user_curve_black_y";
static const char RMD_USER_CURVE_LOW_X[] =				"user_curve_low_x";	
static const char RMD_USER_CURVE_LOW_Y[] =				"user_curve_low_y";	
static const char RMD_USER_CURVE_MID_X[] =				"user_curve_mid_x";	
static const char RMD_USER_CURVE_MID_Y[] =				"user_curve_mid_y";
static const char RMD_USER_CURVE_HIGH_X[] =				"user_curve_high_x";
static const char RMD_USER_CURVE_HIGH_Y[] =				"user_curve_high_y";
static const char RMD_USER_CURVE_WHITE_X[] =			"user_curve_white_x";
static const char RMD_USER_CURVE_WHITE_Y[] =			"user_curve_white_y";
static const char RMD_USER_TIMECODE_PREFERENCE[] =		"user_timecode_preference";		// 0 = edge / run record, 1 = absolute / external timecode, guaranteed to exist
static const char RMD_USER_TITLE_GUIDE_ENABLED[] =		"user_title_guide_enabled";		// 0 = disabled, 1 = enabled
static const char RMD_USER_TITLE_GUIDE_TOPLEFT_X[] =	"user_title_guide_topleft_x";	// float, 0.0 - 1.0, multiply by image width - 1
static const char RMD_USER_TITLE_GUIDE_TOPLEFT_Y[] =	"user_title_guide_topleft_y";	// float, 0.0 - 1.0, multiply by image height - 1
static const char RMD_USER_TITLE_GUIDE_WIDTH[] =		"user_title_guide_width";		// float, 0.0 - 1.0, multiply by image width
static const char RMD_USER_TITLE_GUIDE_HEIGHT[] =		"user_title_guide_height";		// float, 0.0 - 1.0, multiply by image height
static const char RMD_USER_CAMERA_OPERATOR[] =			"user_camera_operator";
static const char RMD_USER_COPYRIGHT[] =				"user_copyright";
static const char RMD_USER_DIRECTOR[] =					"user_director";
static const char RMD_USER_DIRECTOR_OF_PHOTOGRAPHY[] =	"user_director_of_photography";
static const char RMD_USER_EXT_FILTER_1[] =				"user_ext_filter_1";
static const char RMD_USER_EXT_FILTER_2[] =				"user_ext_filter_2";
static const char RMD_USER_EXT_FILTER_3[] =				"user_ext_filter_3";
static const char RMD_USER_EXT_GPS[] =					"user_ext_gps";
static const char RMD_USER_EXT_LUT[] =					"user_ext_lut";
static const char RMD_USER_EXT_PROXY[] =				"user_ext_proxy";
static const char RMD_USER_EXT_TLUT[] =					"user_ext_tlut";
static const char RMD_USER_EXT_UPLOAD_SERVICE[] =		"user_ext_upload_service";
static const char RMD_USER_LOCATION[] =					"user_location";
static const char RMD_USER_PRODUCTION_NAME[] =			"user_production_name";
static const char RMD_USER_SCENE[] =					"user_scene";
static const char RMD_USER_SHOT[] =						"user_shot";
static const char RMD_USER_TAKE[] =						"user_take";
static const char RMD_USER_UNIT[] =						"user_unit";
static const char RMD_WAV_FILENAME[] =					"wav_filename";					// Name of sidecar BWF WAV file, if it was recorded on camera

static const char RMD_WHITE_BALANCE_KELVIN[] =			"white_balance_kelvin";			// Also available through the image processing settings structure
static const char RMD_WHITE_BALANCE_TINT[] =			"white_balance_tint";			// Also available through the image processing settings structure

// Audio metadata items if a clip has embedded audio
static const char RMD_AUDIO_FORMAT[] =					"audio_format";					// 0 = integer, 1 = float (will be auto converted to integer if needed)
static const char RMD_CHANNEL_MASK[] =					"channel_mask";					// audio channel mask, bit 1 = channel 1 enabled ... bit 4 = channel 4 enabled, etc
static const char RMD_SAMPLERATE[] =					"samplerate";					// sample rate, should always be 48000
static const char RMD_SAMPLE_SIZE[] =					"sample_size";					// sample size, should always be 24. This is the actual bits of raw audio data
																						// it might be stored differently like msb aligned in a 32-bit int

// The only metadata items you can count on that exist are framerate (all three), the two start
// timecode values, image width & height, ISO, exposure and user timecode preference

// Per-frame metadata store.
class Metadata
{
public:
	Metadata();
	~Metadata();

	// Functions to retrieve clip metadata through the metadata database
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

private:
	unsigned int reserved1;
	unsigned int reserved2;

	void * reserved;
};

}

#endif
