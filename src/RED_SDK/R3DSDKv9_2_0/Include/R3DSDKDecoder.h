/* R3D SDK library version 9.2 header file. Do *NOT* use this
   header file with any other version of the R3D SDK library!
   
   This header file and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All
   rights reserved. Redistribution of this header is prohibited!
   
   NOTE: GPU decode (decompression), Metal image processing & Extended
         Highlights not supported by R3DDecoder!
*/

#ifndef __R3DDECODER_DYNAMIC_H__
#define __R3DDECODER_DYNAMIC_H__ 1
#include <stdlib.h>
#include <string>
#include <vector>

#include "R3DSDK.h"

namespace R3DSDK
{

	enum R3DStatus
	{
		R3DStatus_Ok = 0,
		R3DStatus_ErrorProcessing = 1,
		R3DStatus_InvalidJobParameter = 2,

        // mode value passed in is not compatible with this SDK or the mode used with the DecodeForGpuSdk call is not compatible
		R3DStatus_InvalidJobParameter_mode = 3,
        
        // pointer is NULL, data is not from DecodeForGpuSdk, R3DSDK and GPUSDK versions are incompatible or buffer is not actually in host memory.
		R3DStatus_InvalidJobParameter_raw_host_mem = 4,
		R3DStatus_InvalidJobParameter_raw_device_mem = 5,
        
        // unsupported pixel type
		R3DStatus_InvalidJobParameter_pixelType = 6,
        
        // Output buffer Size must be non zero.
        // Output buffer must be allocated prior to passing it into the sdk
		R3DStatus_InvalidJobParameter_output_device_mem_size = 7,
		R3DStatus_InvalidJobParameter_output_device_mem = 8,
        
        // Image processing settings ColorVersion was set to ColorVersion1 which is not supported by this SDK
		R3DStatus_InvalidJobParameter_ColorVersion1 = 9,
        
        // null, or unopened clip.
	    R3DStatus_InvalidJobParameter_clip = 10,
        
        // GPU Device did not meet minimum requirements.
		R3DStatus_UnableToUseGPUDevice = 11,
        
        // No GPU Devices were setup on the R3DDecoderOptions class
        R3DStatus_NoGPUDeviceSpecified = 12,
        
        // Error loading R3DSDK dynamic library
        R3DStatus_UnableToLoadLibrary = 13,
        R3DStatus_ParameterUnsupported = 14
	};

	struct OpenCLDeviceInfo
	{
    private:
        unsigned int reserved1;
        unsigned int reserved2;
		
    public:        
        OpenCLDeviceInfo();
        
		// void * values must be set to choose a specific device.
		// platform_id corresponds to a cl_platform_id which is actually _cl_platform_id *
		void * platform_id;
		// device_id corresponds to a cl_device_id which is actually _cl_device_id *
		void * device_id;

		// strings are only used for feedback to sdk user and are not used for any device selection logic.
		char platform_name[256];
		char name[256];
	};

	struct CudaDeviceInfo
	{
    private:
        unsigned int reserved1;
        unsigned int reserved2;
		
    public:
        CudaDeviceInfo();
        
		// pci_bus_id value must be set to choose a specific device bus id.
		// pci_bus_id corresponds to the cudaDeviceProp.pciBusID field
		int pci_bus_id;

		// string name is required to ensure the pci bus id matches the proper device
		char name[256];
	};

	// The decode job structure. This structure needs to be
	// constructed to submit a decode job to decoder.
	// This object should not be reused until the previous call to decode with the job is complete
	struct R3DDecodeJob
	{
    private:
        unsigned int reserved1;
        unsigned int reserved2;
        // should be created using R3DDecoder::CreateDecodeJob
        // and released with R3DDecoder::ReleaseDecodeJob
        R3DDecodeJob();
        
    public:
		// Frame to decode for which clip. This cannot be NULL.
		Clip *						clip;

		size_t						videoTrackNo;			// 0 = main (A) track, 1 = EPIC/Scarlet-X higlight protection track 2 (X track)
															// ignored when doing HDRx blending
		size_t						videoFrameNo;

		// item is the pointer to the original item as submitted in the Decode() call
		typedef void (*DecodeCallback)(R3DDecodeJob * item, R3DStatus decodeStatus);

		// Callback to call when frame is available in the specified output buffer
		// You will receive a call for each frame submitted and these calls will
		// come in on seperate threads, potentially at the same time.
		// SPEND AS LITTLE TIME AS POSSIBLE IN THE CALLBACK TO NOT HOLD UP PROCESSING OTHER FRAMES!
		DecodeCallback				callback;
		void *						privateData;			// available for your use as you see fit, R3D SDK will not touch this variable!

		// Resolution/speed to decode the image at. This will also 
		// influence how much memory is needed for the output buffer
		VideoDecodeMode				mode;

	    
		// Which pixel type to decode the image in. See the VideoPixelType
		// enum for the available options. 
		// using an unsupported pixeltype will result in an InvalidPixelType error code being returned.
		// Supported VideoPixelTypes:
		//			PixelType_16Bit_RGB_Planar
		//			PixelType_16Bit_RGB_Interleaved
		//			PixelType_10Bit_DPX_MethodB
		//			PixelType_8Bit_BGR_Interleaved
		//			PixelType_HalfFloat_RGB_Interleaved
		//			PixelType_HalfFloat_RGB_ACES_Int
		VideoPixelType				pixelType;

		// Pointer to the buffer to store the image in. This cannot be
		// NULL otherwise the decode will fail.
		void *						outputBuffer;

		// Number of bytes in a row
		size_t						bytesPerRow;

		// Size ouf the output buffer
		size_t						outputBufferSize;
		
		// Image processing settings to apply to the decode. This cannot be NULL.
		ImageProcessingSettings		*imageProcessingSettings;
        
		Metadata *outputFrameMetadata;
	};


	class R3DDecoderOptions
	{
	public:
		virtual ~R3DDecoderOptions() {}

		// creates an instance of an R3DDecoderOptions.
		static R3DStatus CreateOptions(R3DDecoderOptions **options_out);

	    // releases options created by CreateOptions
		static void ReleaseOptions(R3DDecoderOptions *options);

        // The set routines on this class are available to tweak the performance characteristics of the R3DDecoder.
        // The SDK will automatically choose sane defaults based upon which devices are chosen with the useDevice call, however if you need to restrict the SDK for a specific use case you can use the set routines to do so.
        // The defaults the SDK chooses are usually core count - 1 for threads, and min(device memory * 0.75, concurrent GPU Frames * 1GB) for GPUs, for host memory the value is by default set to concurrentImageCount * 512MB,  note if the device of the host actually runs out of memory the out of memory status will be returned from either the callback or the decode call, based upon when it occurs.  Memory pools do not pre-allocate the memory.
        
	    // sets the folder for temporary files and compiled kernels.
	    // if set to empty string or not called no temporary folder will be used.
        // On some older boards the kernel compilation may take upwards of 5 seconds, this is present to mitigate that and make it a one time occurance instaed of per application run.
        // On modern boards the compilation time is hardly noticable.
		R3DStatus setScratchFolder(std::string path);
		
		// 0 = use (cores - 1)
        // This is the number of active CPU threads used for decompression of the footage.
        // This does not have a 1:1 ratio to the number of frames being processed at a time, a single frame in most cases will be split to operate on multiple of these task threads.
        // We recommend 1 - core count to prevent full CPU usage keeping you from interacting with the computer while processing.
		R3DStatus setDecompressionThreadCount(size_t decompressionThreadCount);

		// 0 = use (cores - 1)
        // This number determines the how many threads will be used to manage the frame(s) lifetime, from your request all the way to the complete callback.
        // When you call decode, your request gets verified, prepared and then queued until one of these threads is available.
        // For each of these threads some buffers will need to be allocated in host memory to handle the decompression step.
		R3DStatus setConcurrentImageCount(size_t imageThreadCount);

		// minimum 1024 recommended 4096 or higher
        // This value is used for allocations of Host memory, the value passed into here will need to scale for the value passed to setConcurrentImageCount.
		R3DStatus setMemoryPoolSize(size_t memoryPoolMbs);

		// minimum 1024 recommended 4096 or higher
        // The memory pool is used to keep larger buffers on the device around to be re-used between frames.
        // All of our GPU buffer types come from here, pinned, and temp processing buffers, if this is not high enough for the value passed to setGPUConcurrentFrameCount you will not be running as optimal as possible, though it should not be higher than what the graphics card has available.
		R3DStatus setGPUMemoryPoolSize(size_t memoryPoolMbs);
		
		// Sets number of cudaStreams or OpenCL cl_command_queue(s).
		// minimum 1 - recommended 3 (if GPU has enough RAM)
        // This value determines how many frames we will attempt to process on the GPU at a time, a frame consists of HostToDevice/ run kernels/ DeviceToHost.
        // By increasing the number of frames you directly 1:1 increase memory requirement to process on the card.
        // The memory requirement to process a single frame on the GPU is not light as RED deals with very large frame sizes, there are also several other utility buffers needed when processing the larger of them are as follows: upload pinned buffer, temp processing buffer, download pinned buffer.
        // Due to the limitation of hardware only overlapping upto 3 of these actions, there is not much benefit to this number going above 3.
        // More about the hardware limitation: Modern GPU Devices support at most processing of 3 actions at once by overlapping actions in different streams/command queues.
        // The types of actions are: HostToDevice/Kerenel/DeviceToHost.
        // The high end quadros can do one of each overlapped as they have 2 transfer engines and 1 kernel engine, where the Geforce line can only do 2 of them at a time meaning they cannot upload and download concurrently.
		R3DStatus setGPUConcurrentFrameCount(size_t  numConcurrentFrames);

	    // returns a list of opencl devices compatible with SDK
		static R3DStatus GetOpenCLDeviceList(std::vector<OpenCLDeviceInfo> &devices);

	    // sets the opencl device to be used by the decoder
	    // cuda and opencl are mutually exclusive, you cannot combine usage of cuda and opencl on the same options instance
		R3DStatus useDevice(OpenCLDeviceInfo deviceInfo);

	    // returns a list of opencl devices compatible with SDK
		static R3DStatus GetCudaDeviceList(std::vector<CudaDeviceInfo> &devices);

	    // sets the cuda device to be used by the decoder
	    // cuda and opencl are mutually exclusive, you cannot combine usage of cuda and opencl on the same options instance
		R3DStatus useDevice(CudaDeviceInfo deviceInfo);

		void *reserved;
	private:
		R3DDecoderOptions(){}
	};

	class R3DDecoder
	{
	protected:
		R3DDecoder();
		virtual ~R3DDecoder();

	public:
	    // creates an instance of an R3DDecoder based upon the options object.
		static R3DStatus CreateDecoder(R3DDecoderOptions *options, R3DDecoder **decoder_out);

	    // releases r3ddecoders created by CreateDecoder
		static void ReleaseDecoder(R3DDecoder *decoder);

		static R3DStatus CreateDecodeJob(R3DDecodeJob **job_out);
		
		static void ReleaseDecodeJob(R3DDecodeJob *job);
	    // decodes a frame based upon the job options.
		// The job object should not be reused or disposed until the previous call to decode with the job is complete
		R3DStatus decode(R3DDecodeJob *job);
	private:
		void *reserved;
	};
}

#endif
