/* R3D SDK library version 9.2 header file. Do *NOT* use this
   header file with any other version of the R3D SDK library!
   
   This header file and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All
   rights reserved. Redistribution of this header is prohibited!
*/

#ifndef R3DSDKMETAL_H
#define R3DSDKMETAL_H

#include <Metal/Metal.h>
#include "R3DSDK.h"

namespace R3DSDK
{
	
// allows device memory management to be handled in calling application
// this can enhance performance when resources are being cached
// default instance of the class makes the SDK call Metal directly
class EXT_METAL_API
{
private:
    unsigned int reserved1;
    unsigned int reserved2;
        
public:
    EXT_METAL_API();
        
    id<MTLBuffer> (*createMTLBuffer)(id<MTLDevice> device, size_t size, void * hostPtr, MTLResourceOptions mode, int &error);
    id<MTLTexture> (*createMTLTexture)(id<MTLDevice> device, MTLTextureDescriptor * descriptor, int &error);
        
    void (*releaseMTLBuffer)(id<MTLBuffer> buffer);
    void (*releaseMTLTexture)(id<MTLTexture> texture);
};

class DebayerMetalJob
{
private:
    unsigned int reserved1;
    unsigned int reserved2;

    DebayerMetalJob();
    ~DebayerMetalJob();
public:
    // returns the size in bytes of the result frame that the debayer job passed in will produce.
    static size_t ResultFrameSize( const DebayerMetalJob &debayerJob );

    // This will ensure all objects used for the frame are disposed of.
    // This call will block until the debayer on the queue executes, if the debayer has already executed no blocking will occur
    // This function must be called at a point after passing the DebayerMetalJob into REDMetal::processAsync
    // This will automatically be called upon destruction the DebayerMetalJob if it is not manually called earlier
    void completeAsync();
    
	// NOTE: This pointer + MTLBuffer & their contents must remain valid until completeAsync() or process() returns!
    void *raw_host_mem; 				// sized from r3dsdk buffer
    id<MTLBuffer> raw_device_mem;		// sized from r3dsdk buffer

    size_t output_device_mem_size;		// size of the output buffer
										// it must be a minimum of ResultFrameSize(job)

    id<MTLBuffer> output_device_mem; 	// result/output buffer on the GPU
										//    object of the size output_device_mem_size
										//    -- do not use output_device_image when using this field

    id<MTLTexture> output_device_image; // alternate result/output texture on the GPU (16-bit uint and float supported)
                                        // use this field if you want the result image written to an MTLTexture
                                        // -- do not use output_device_mem when using this field
							  
    VideoDecodeMode mode;

	// Image Processing Settings. This cannot be NULL.
    ImageProcessingSettings *imageProcessingSettings;

    // Supported Types are:
    //  PixelType_16Bit_RGB_Interleaved 
    //  PixelType_HalfFloat_RGB_Interleaved
    //  PixelType_HalfFloat_RGB_ACES_Int
    VideoPixelType pixelType;
    
    // Enables batch processing of frames - frames added to a single batch will all complete together when flush() is called 
	// True to add frame to the batch, queuing up the work
	// False to add frame to the batch, and then complete the whole batch
    // Leaving the field False will cause frames to be processed individually
	// Recommended batch size: 2 for laptops, 4 for iMac Pro, 8 for Mac Pro (2019)
    bool batchMode;
    
    // Provides a callback that will be called when the job completes
    // This can be used instead of calling completeAsync
    // Can be NULL
    void (* debayerJobCallback)(DebayerMetalJob * job);

private:
    void *reserved;
};

class REDMetal
{
public:
    enum Status
    {
        Status_Ok = 0,
        Status_ErrorProcessing = 1,
        Status_InvalidJobParameter = 2,
        
        // mode value passed in is not compatible with this SDK or the mode used with the DecodeForGpuSdk call is not compatible
        Status_InvalidJobParameter_mode = 3,	// mode is not compatible with this SDK or the mode used with the DecodeForGpuSdk call
        
        // pointer is NULL, data is not from DecodeForGpuSdk, R3DSDK and GPUSDK versions are incompatible or buffer is not actually in host memory.
        Status_InvalidJobParameter_raw_host_mem = 4,
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
		
		Status_InvalidAPIObject = 13
    };

	DebayerMetalJob *createDebayerJob();

    void releaseDebayerJob(DebayerMetalJob *job);

	// api parameter can be an empty instance of EXT_METAL_API in which case the
	// SDK calls Metal directly for resource allocation. If pointers are set then
	// the application will handle memory management and caching for increased
	// performance. This is critical to get maximum GPU decode performance.
	REDMetal(EXT_METAL_API & api);
	~REDMetal();

    // checks to see if a context and command queue are compatible with REDMetal
    // This call may take several seconds on slower cards
    Status checkCompatibility(id<MTLCommandQueue> queue, int &error);
    
    // processes a frame.
    // Thread safe blocking call
    // This function will enqueue the Metal commands and wait for the processing to complete
    //  	to ensure memory buffers are not released prior to completion of kernel
    Status process(id<MTLCommandQueue> queue, DebayerMetalJob *debayerJob, int &error);

    // enqueues all red processing on the current command queue. NOTE: All calls to processAsync
	// should be done using a single MTLCommandQueue to avoid potential MTLCommandBuffer failures
    // once processAsync has been called you will need to call debayerJob->completeAsync
    //      This will wait for commands enqueued by this sdk for the current frame to complete and then releases any additional resources.
    //      If the commands have already completed there is no wait involved
    Status processAsync(id<MTLCommandQueue> queue, DebayerMetalJob *debayerJob, int &error);
    
    // flushes the current batch job that has been queued for the current MTLCommandQueue
    // after this has been called the jobs in the current batch will be processed in the order they were submitted
    // completeAsync() will return after each job in the batch completes
    // if debayerJobCallback has been set, the callback function will be called after each job completes
    Status flush(id<MTLCommandQueue> queue, int &error);

private:
    void *reserved;
};

}

#endif

