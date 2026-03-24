/* R3D SDK library version 9.2 header file. Do *NOT* use this
   header file with any other version of the R3D SDK library!
   
   This header file and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All
   rights reserved. Redistribution of this header is prohibited!
*/

#ifndef R3DSDKSTREAM_H
#define R3DSDKSTREAM_H

namespace R3DSDK {

enum VerifyStatus
{
    /**
     * Packet is verified.  OK to pass on to WritePacketData(...)
     */
    VS_OK = 0,

    /**
     * Packet should be dropped.  Do not pass to WritePacketData(...)
     */
    VS_DROP = 1,

    /**
     * Missing callback to send an ACK.  Do not pass to WritePacketData(...)
     */
    VS_MISSING_CALLBACK = 2,

    /**
     * Error communicating with dynamic library.
     */
	VS_INTERNAL_ERROR = 3,

    /**
     * This (version of the) SDK cannot handle this stream format.
     */
	VS_UNSUPPORTED_STREAM_FORMAT = 4
};

// Class that will help in filtering the incoming RCP stream from the camera
// for tethering. 
class CameraStream
{
public:
	/**
	* Callback method for notifying the API user that an ACK packet must be
	* sent to the camera.
	*
	* ackData will be destroyed when this function returns.  Send the data
	* to the camera immediately.
	*/
	typedef void (*AckCallback)(void *userData, const unsigned char *ackData, unsigned int len);

	CameraStream(AckCallback ackCallback, void *ackUserData);
	~CameraStream();

	// call from network thread with packet received, try to do nothing
	// else in this thread to make sure no packets are dropped! If return
	// status is VS_OK pass the packet to an instance of R3DStream::WritePacketData()
	VerifyStatus ProcessRdpPacket(const void * packet, size_t packetLength);

private:
	void * reserved;
};

// Class to stream the incoming data to an R3D clip. A new instance should be created
// for each individual clip so the state can be properly tracked and clips correctly
// closed. Can be used standalone or in combination with the DecodeStream class.
class R3DStream
{
public:
	// create new instance, 
	R3DStream(const char * outputPath, unsigned int cameraReelId, unsigned int cameraClipId);
	~R3DStream();

	// call to close current clip and create a new one mid-stream. This is useful to split
	// the stream up in to multiple clips without having gaps in-between them. Returns false
	// if supplied parameters are invalid or current clip could not be closed.
	bool SetOutputClip(unsigned int cameraReelId, unsigned int cameraClipId);

	// call from processing thread if ProcessRdpPacket() returned true
	// returns CSFrameAdded if a block is complete, otherwise returns CSDone or error code.
	// output parameters are filled when CSFrameAdded is returned.
	CreateStatus WritePacketData(const void * packet, size_t packetLength, Metadata * metadataToFill = NULL);
	CreateStatus WritePacketData(const void * packet, size_t packetLength, bool & isDroppedFrame, Metadata * metadataToFill = NULL);

private:
	void * reserved;
};

// Class to decode the incoming data for preview. Decoding must happen on a separate thread
// so incoming data from the camera can still be processed. Can be used standalone or in
// combination with the R3DStream class.
class DecodeStream
{
public:
	DecodeStream();
	~DecodeStream();

	// call from processing thread if ProcessRdpPacket() returned true
	// returns CSFrameAdded if a block is complete, otherwise returns CSDone or error code
	CreateStatus ProcessPacketData(const void * packet, size_t packetLength, bool & isDroppedFrame);

	// call this from a separate decode thread when ProcessPacketData() returns CSFrameAdded.
	// Use lower resolution decodes to speed up the decode and allow for smoother preview.
	DecodeStatus DecodeLastReceivedFrame(const VideoDecodeJob & decodeJob);

private:
	void * reserved;
};

#ifdef RCP_TYPES_PUBLIC_H
// This class can be used to update the UI state on the camera while streaming
// data over the tether. Camera should be told if the stream is being saved or
// not and the reel & clip ID that the clip is being saved under.
class CameraUIState
{
public:
	/**
	 * Function pointer to the rcp_set_int() function.
	 *
	 * User can pass us this function pointer to allow us to set clip number, record indicator color, etc.
	 */
	typedef rcp_error_t (*RcpSetInt)(rcp_camera_connection_t *con, rcp_param_t id, int val);
	
	/**
	 * Function pointer to the rcp_set_uint() function
	 */
	typedef rcp_error_t (*RcpSetUInt)(rcp_camera_connection_t *con, rcp_param_t id, unsigned int val);

	CameraUIState(RcpSetInt rcpsi, RcpSetUInt rcpsui, rcp_camera_connection_t *con);
	~CameraUIState();

	// set camera UI record state
	// returns RCP_ERROR_BAD_PARAM if no rcp_set_uint() function pointer or camera connection was set
	rcp_error_t SetUIRecordState(bool showRecordState) const;

	// let camera know the total space the volume being written to
	rcp_error_t SetTotalStorageSpace(unsigned int megabytesTotalStorage) const;

	// let camera know how much space out of the total storage space is in use
	rcp_error_t SetStorageSpaceLeft(unsigned int megabytesStorageLeft) const;

	// set UI reel ID
	// reelId must be in range 1 - 999
	rcp_error_t SetReelId(unsigned int reelId) const;

	// set UI clip ID
	// clipId must be in range 1 - 999
	rcp_error_t SetClipId(unsigned int clipId) const;
	
private:
	void * reserved;
};
#endif

class R3DStream2
{
public:
	// outputPath must be a folder
	// cameraReelId must be in range 1 - 999
	// cameraClipId must be in range 1 - 999
	// maxFilesizeInGB if set must be in range 4 - 1024 (1 TB). If not set, camera file size is used.
	R3DStream2(const char * outputPath, unsigned int cameraReelId, unsigned int cameraClipId, unsigned int maxFilesizeInGB = 0);
	~R3DStream2();

	// returns 0 if data is complete and WriteData() should be called, -1 if input parameters are invalid,
	// or otherwise returns the additional bytes that still need to be read for the frame to be complete
	// may update currentIndex
	int BytesToRead(const void * data, int & currentIndex) const;

	// returns CSStarted when a clip is created
	// returns CSFrameAdded when data was succesfully processed after clip was created
	// returns CSDone when a clip has been closed
	// otherwise returns an error
	// sets currentIndex to 0 when data was processed
	// isDroppedFrame is set to true if the frame was dropped by the camera because it was too far behind
	CreateStatus WriteData(void * data, int & currentIndex, bool & isDroppedFrame, Metadata * metadataToFill = NULL);

private:
	void * reserved;
};

}

#endif

