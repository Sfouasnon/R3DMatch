/* R3D SDK library version 9.2 header file. Do *NOT* use this
   header file with any other version of the R3D SDK library!
   
   This header file and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All
   rights reserved. Redistribution of this header is prohibited!
*/

#ifndef __SDK_DYNAMIC_REDCUDA_H__
#define __SDK_DYNAMIC_REDCUDA_H__ 1
#include <cuda_runtime.h>
#include "R3DSDK.h"

namespace R3DSDK
{
#if defined(_WIN32)
#define CUDA_API_ENTRY
#define CUDA_API_CALL     __stdcall
#define CUDA_CALLBACK     __stdcall
#else
#define CUDA_API_ENTRY
#define CUDA_API_CALL
#define CUDA_CALLBACK
#endif
    

class EXT_CUDA_API
{
private:
    unsigned int reserved1;
    unsigned int reserved2;
	
public:
    EXT_CUDA_API();
    cudaError_t (CUDA_API_CALL *cudaFree)( void* devPtr );
    cudaError_t (CUDA_API_CALL *cudaFreeArray)( cudaArray_t array );
    cudaError_t (CUDA_API_CALL *cudaFreeHost)( void* ptr );
    cudaError_t (CUDA_API_CALL *cudaFreeMipmappedArray)( cudaMipmappedArray_t mipmappedArray );
    cudaError_t (CUDA_API_CALL *cudaHostAlloc)( void** pHost, size_t size, unsigned int  flags );
    cudaError_t (CUDA_API_CALL *cudaMalloc)( void** devPtr, size_t size );
    cudaError_t (CUDA_API_CALL *cudaMalloc3D)( cudaPitchedPtr* pitchedDevPtr, cudaExtent extent );
    cudaError_t (CUDA_API_CALL *cudaMalloc3DArray)( cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  flags );
    cudaError_t (CUDA_API_CALL *cudaMallocArray)( cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height, unsigned int  flags );
    cudaError_t (CUDA_API_CALL *cudaMallocHost)( void** ptr, size_t size );
    cudaError_t (CUDA_API_CALL *cudaMallocMipmappedArray)( cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  numLevels, unsigned int  flags );
    cudaError_t (CUDA_API_CALL *cudaMallocPitch)( void** devPtr, size_t* pitch, size_t width, size_t height );
};


class DebayerCudaJob
{
private:
    unsigned int reserved1;
    unsigned int reserved2;

    DebayerCudaJob();
	~DebayerCudaJob();
	
public:
	// returns the size in bytes of the result frame that the debayer job passed in will produce.
	static size_t ResultFrameSize( const DebayerCudaJob *debayerJob );

	// This will ensure all objects used for the frame are disposed of.
	// This call will block until the debayer on the stream executes, if the debayer has already executed no blocking will occur
	// This function must be called at a point after passing the debayerCudaJob into REDCuda::processAsync
	// This will automatically be called upon destruction the DebayerCudaJob if it is not manually called earlier
	void completeAsync();
    
	// NOTE: These pointers & their content must remain valid until completeAsync() or process() returns!
	void *raw_host_mem;   // Result buffer from R3D ASYNC SDK
	void *raw_device_mem; // Same data as raw_host_mem except already on the GPU

	size_t output_device_mem_size; // size of the output buffer
									  // if 0 an error will be returned
									  // it must be a minimum of ResultFrameSize(job)

	void *output_device_mem;  // result/output buffer on the GPU
							  // this must point to device memory
							  //	of the size output_device_mem_size
	
	VideoDecodeMode mode;
	
	// Image Processing Settings. This cannot be NULL.
	ImageProcessingSettings *imageProcessingSettings;
	
	// Supported Types are: 
	//	PixelType_16Bit_RGB_Interleaved	
	//  PixelType_HalfFloat_RGB_Interleaved
	//  PixelType_HalfFloat_RGB_ACES_Int
	VideoPixelType pixelType;

private:
	void *reserved;
};

class REDCuda
{
public:
	enum Status
	{
		Status_Ok = 0,
		Status_ErrorProcessing = 1,
		Status_InvalidJobParameter = 2,
        
        // mode value passed in is not compatible with this SDK or the mode used with the DecodeForGpuSdk call is not compatible
		Status_InvalidJobParameter_mode = 3,
        
        // pointer is NULL, data is not from DecodeForGpuSdk, R3DSDK and GPUSDK versions are incompatible or buffer is not actually in host memory.
		Status_InvalidJobParameter_raw_host_mem = 4,
        
        // Raw device mem was NULL
		Status_InvalidJobParameter_raw_device_mem = 5,
        
        // unsupported pixel type
		Status_InvalidJobParameter_pixelType = 6,
        
        // Output buffer Size must be non zero.
        // Output buffer must be allocated prior to passing it into the sdk
		Status_InvalidJobParameter_output_device_mem_size = 7,
		Status_InvalidJobParameter_output_device_mem = 8,
        
        // Image processing settings ColorVersion was set to ColorVersion1 which is not supported by this SDK
		Status_InvalidJobParameter_ColorVersion1 = 9,
        
        // GPU Device did not meet minimum requirements.
		Status_UnableToUseGPUDevice = 10,
        
        // Error loading R3DSDK dynamic library
        Status_UnableToLoadLibrary = 11,
        Status_ParameterUnsupported = 12,
        
        // deviceId must be an index between 0 and the number of devices on the system minus one
        Status_InvalidJobParameter_deviceId = 13
	};

	DebayerCudaJob *createDebayerJob();

    void releaseDebayerJob(DebayerCudaJob *job);

	// EXT_CUDA_API must be bound to your desired implementation of Cuda.	
	REDCuda(EXT_CUDA_API &cuda_api_functions);

	~REDCuda();

	// checks to see if a device and stream queue are compatible with REDCuda
	// This call may take several seconds on slower cards
	Status checkCompatibility(int deviceId, cudaStream_t stream, cudaError_t &cuda_error);

	// processes a frame.
	// Thread safe blocking call
	// This function will enqueue the cuda kernels and wait for the processing to complete
	//		to ensure memory buffers are not released prior to completion of kernel
	//		deviceId and stream are used as tokens for internal object mapping.  
	//		Be sure you have already done setCudaDevice on the thread using the passed in deviceId prior to calling process
	Status process(int deviceId, cudaStream_t stream, DebayerCudaJob *debayerJob, cudaError_t &cuda_error);
	
	// enqueues all red processing on the current stream
	// once processAsync has been called you will need to call debayerJob->completeAsync
	//		This will wait for commands enqueued by this sdk for the current frame to complete and then releases any additional resources. 
	//		If the commands have already completed there is no wait involved
	Status processAsync(int deviceId, cudaStream_t stream, DebayerCudaJob *debayerJobAsync, cudaError_t &cuda_error);

private:
	void *reserved;
};

}

#endif
