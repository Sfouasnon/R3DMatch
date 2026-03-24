/* R3D SDK library version 9.2 header file. Do *NOT* use this
   header file with any other version of the R3D SDK library!
   
   This header file and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All
   rights reserved. Redistribution of this header is prohibited!
*/

#ifndef __SDK_REDCL_H__
#define __SDK_REDCL_H__ 1
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/opencl.h>
#endif
#include "R3DSDK.h"

#ifndef  cl_buffer_create_type
typedef cl_uint cl_buffer_create_type;
#endif

namespace R3DSDK
{

    
#if defined(_WIN32)
#define CL_API_ENTRY
#define CL_API_CALL     __stdcall
#define CL_CALLBACK     __stdcall
#else
#define CL_API_ENTRY
#define CL_API_CALL
#define CL_CALLBACK
#endif
    
class EXT_OCLAPI_1_1
{
private:
    unsigned int reserved1;
    unsigned int reserved2;
public:
    
    EXT_OCLAPI_1_1();
    //OCL 1.0 interface
    cl_int (CL_API_CALL *clSetKernelArg)(cl_kernel    /* kernel */,
                                         cl_uint      /* arg_index */,
                                         size_t       /* arg_size */,
                                         const void * /* arg_value */);
    cl_int (CL_API_CALL *clFlush)(cl_command_queue /* command_queue */);
    cl_int (CL_API_CALL *clFinish)(cl_command_queue /* command_queue */);
    cl_int (CL_API_CALL *clEnqueueCopyImage)(cl_command_queue     /* command_queue */,
                                             cl_mem               /* src_image */,
                                             cl_mem               /* dst_image */,
                                             const size_t *       /* src_origin[3] */,
                                             const size_t *       /* dst_origin[3] */,
                                             const size_t *       /* region[3] */,
                                             cl_uint              /* num_events_in_wait_list */,
                                             const cl_event *     /* event_wait_list */,
                                             cl_event *           /* event */);
    
    cl_context (CL_API_CALL *clCreateContext)(const cl_context_properties * /* properties */,
                                              cl_uint                 /* num_devices */,
                                              const cl_device_id *    /* devices */,
                                              void (CL_CALLBACK * /* pfn_notify */)(const char *, const void *, size_t, void *),
                                              void *                  /* user_data */,
                                              cl_int *                /* errcode_ret */);
    cl_command_queue (CL_API_CALL *clCreateCommandQueue)(cl_context                     /* context */,
                                                         cl_device_id                   /* device */,
                                                         cl_command_queue_properties    /* properties */,
                                                         cl_int *                       /* errcode_ret */);
    cl_sampler (CL_API_CALL *clCreateSampler)(cl_context          /* context */,
                                              cl_bool             /* normalized_coords */,
                                              cl_addressing_mode  /* addressing_mode */,
                                              cl_filter_mode      /* filter_mode */,
                                              cl_int *            /* errcode_ret */);
    cl_kernel (CL_API_CALL *clCreateKernel)(cl_program      /* program */,
                                            const char *    /* kernel_name */,
                                            cl_int *        /* errcode_ret */);
    cl_mem (CL_API_CALL *clCreateBuffer)(cl_context   /* context */,
                                         cl_mem_flags /* flags */,
                                         size_t       /* size */,
                                         void *       /* host_ptr */,
                                         cl_int *     /* errcode_ret */);
    
    cl_program (CL_API_CALL *clCreateProgramWithSource)(cl_context        /* context */,
                                                        cl_uint           /* count */,
                                                        const char **     /* strings */,
                                                        const size_t *    /* lengths */,
                                                        cl_int *          /* errcode_ret */);
    
    cl_program (CL_API_CALL *clCreateProgramWithBinary)(cl_context /* context */,
                                                        cl_uint /* num_devices*/,
                                                        const cl_device_id * /* device_list */,
                                                        const size_t * /* lengths */,
                                                        const unsigned char ** /*binaries*/,
                                                        cl_int * /*binary_status*/,
                                                        cl_int * /*errcode_ret*/);
    cl_int (CL_API_CALL *clReleaseEvent)(cl_event /*event*/);
    cl_int (CL_API_CALL *clReleaseSampler)(cl_sampler /* sampler */);
    cl_int (CL_API_CALL *clReleaseKernel)(cl_kernel   /* kernel */);
    cl_int (CL_API_CALL *clReleaseMemObject)(cl_mem /* memobj */);
    cl_int (CL_API_CALL *clReleaseProgram)(cl_program /* program */);
    cl_int (CL_API_CALL *clReleaseContext)(cl_context /* context */);
    cl_int (CL_API_CALL *clReleaseCommandQueue)(cl_command_queue /* command_queue */);
    
    cl_int (CL_API_CALL *clGetPlatformInfo)(cl_platform_id   /* platform */,
                                            cl_platform_info /* param_name */,
                                            size_t           /* param_value_size */,
                                            void *           /* param_value */,
                                            size_t *         /* param_value_size_ret */);
    cl_int (CL_API_CALL *clGetDeviceIDs)(cl_platform_id   /* platform */,
                                         cl_device_type   /* device_type */,
                                         cl_uint          /* num_entries */,
                                         cl_device_id *   /* devices */,
                                         cl_uint *        /* num_devices */);
    cl_int (CL_API_CALL *clGetPlatformIDs)(cl_uint          /* num_entries */,
                                           cl_platform_id * /* platforms */,
                                           cl_uint *        /* num_platforms */);
    cl_int (CL_API_CALL *clGetDeviceInfo)(cl_device_id    /* device */,
                                          cl_device_info  /* param_name */,
                                          size_t          /* param_value_size */,
                                          void *          /* param_value */,
                                          size_t *        /* param_value_size_ret */);
    
    cl_int (CL_API_CALL *clGetContextInfo) (cl_context /* context */,
                                            cl_context_info /* param_name */,
                                            size_t /* param_value_size */,
                                            void * /* param_value */,
                                            size_t * /* param_value_size_ret */);
    
    cl_int (CL_API_CALL *clGetImageInfo)(cl_mem           /* image */,
                                         cl_image_info    /* param_name */,
                                         size_t           /* param_value_size */,
                                         void *           /* param_value */,
                                         size_t *         /* param_value_size_ret */);
    cl_int (CL_API_CALL *clGetProgramBuildInfo)(cl_program            /* program */,
                                                cl_device_id          /* device */,
                                                cl_program_build_info /* param_name */,
                                                size_t                /* param_value_size */,
                                                void *                /* param_value */,
                                                size_t *              /* param_value_size_ret */);
    cl_int (CL_API_CALL *clGetProgramInfo)(cl_program /* program */,
                                           cl_program_info /* param_name */,
                                           size_t /* param_value_size */,
                                           void * /* param_value */,
                                           size_t * /* param_value_size_ret */);
    
    cl_int (CL_API_CALL *clGetKernelWorkGroupInfo)(cl_kernel                  /* kernel */,
                                                   cl_device_id               /* device */,
                                                   cl_kernel_work_group_info  /* param_name */,
                                                   size_t                     /* param_value_size */,
                                                   void *                     /* param_value */,
                                                   size_t *                   /* param_value_size_ret */);
    
    cl_int (CL_API_CALL *clBuildProgram)(cl_program           /* program */,
                                         cl_uint              /* num_devices */,
                                         const cl_device_id * /* device_list */,
                                         const char *         /* options */,
                                         void (CL_CALLBACK *  /* pfn_notify */)(cl_program /* program */, void * /* user_data */),
                                         void *               /* user_data */);
    
    cl_int (CL_API_CALL *clEnqueueWriteBuffer)(cl_command_queue   /* command_queue */,
                                               cl_mem             /* buffer */,
                                               cl_bool            /* blocking_write */,
                                               size_t             /* offset */,
                                               size_t             /* size */,
                                               const void *       /* ptr */,
                                               cl_uint            /* num_events_in_wait_list */,
                                               const cl_event *   /* event_wait_list */,
                                               cl_event *         /* event */);
    cl_int (CL_API_CALL *clEnqueueReadBuffer)(cl_command_queue    /* command_queue */,
                                              cl_mem              /* buffer */,
                                              cl_bool             /* blocking_read */,
                                              size_t              /* offset */,
                                              size_t              /* size */,
                                              void *              /* ptr */,
                                              cl_uint             /* num_events_in_wait_list */,
                                              const cl_event *    /* event_wait_list */,
                                              cl_event *          /* event */);
    cl_int (CL_API_CALL *clEnqueueCopyBuffer)(cl_command_queue    /* command_queue */,
                                              cl_mem              /* src_buffer */,
                                              cl_mem              /* dst_buffer */,
                                              size_t              /* src_offset */,
                                              size_t              /* dst_offset */,
                                              size_t              /* size */,
                                              cl_uint             /* num_events_in_wait_list */,
                                              const cl_event *    /* event_wait_list */,
                                              cl_event *          /* event */);
    cl_int (CL_API_CALL *clEnqueueCopyBufferToImage)(cl_command_queue    /* command_queue */,
                                                     cl_mem              /* src_buffer */,
                                                     cl_mem              /* dst_buffer */,
                                                     size_t              /* src_offset */,
                                                     const size_t []              /* dst_offset */,
                                                     const size_t []             /* region */,
                                                     cl_uint             /* num_events_in_wait_list */,
                                                     const cl_event *    /* event_wait_list */,
                                                     cl_event *          /* event */);
    cl_int (CL_API_CALL *clEnqueueWriteImage)(cl_command_queue    /* command_queue */,
                                              cl_mem              /* image */,
                                              cl_bool             /* blocking_write */,
                                              const size_t *      /* origin[3] */,
                                              const size_t *      /* region[3] */,
                                              size_t              /* input_row_pitch */,
                                              size_t              /* input_slice_pitch */,
                                              const void *        /* ptr */,
                                              cl_uint             /* num_events_in_wait_list */,
                                              const cl_event *    /* event_wait_list */,
                                              cl_event *          /* event */);
    cl_int (CL_API_CALL *clEnqueueNDRangeKernel)(cl_command_queue /* command_queue */,
                                                 cl_kernel        /* kernel */,
                                                 cl_uint          /* work_dim */,
                                                 const size_t *   /* global_work_offset */,
                                                 const size_t *   /* global_work_size */,
                                                 const size_t *   /* local_work_size */,
                                                 cl_uint          /* num_events_in_wait_list */,
                                                 const cl_event * /* event_wait_list */,
                                                 cl_event *       /* event */);
    void * (CL_API_CALL *clEnqueueMapBuffer)(cl_command_queue /* command_queue */,
                                             cl_mem           /* buffer */,
                                             cl_bool          /* blocking_map */,
                                             cl_map_flags     /* map_flags */,
                                             size_t           /* offset */,
                                             size_t           /* size */,
                                             cl_uint          /* num_events_in_wait_list */,
                                             const cl_event * /* event_wait_list */,
                                             cl_event *       /* event */,
                                             cl_int *         /* errcode_ret */);
    cl_int (CL_API_CALL *clEnqueueUnmapMemObject)(cl_command_queue /* command_queue */,
                                                  cl_mem           /* memobj */,
                                                  void *           /* mapped_ptr */,
                                                  cl_uint          /* num_events_in_wait_list */,
                                                  const cl_event *  /* event_wait_list */,
                                                  cl_event *        /* event
                                                                     * */);
    cl_int (CL_API_CALL *clWaitForEvents)(cl_uint /* num_events */,
                                          const cl_event * /* event_liist */ );
    
    cl_int (CL_API_CALL *clEnqueueBarrier)(cl_command_queue /* command queue */);
    
    cl_int (CL_API_CALL *clEnqueueMarker)(cl_command_queue /* command queue */,
                                          cl_event * /*event*/);
    
    //OCL 1.1 subset
    cl_mem (CL_API_CALL *clCreateImage2D)(cl_context              /* context */,
                                          cl_mem_flags            /* flags */,
                                          const cl_image_format * /* image_format */,
                                          size_t                  /* image_width */,
                                          size_t                  /* image_height */,
                                          size_t                  /* image_row_pitch */,
                                          void *                  /* host_ptr */,
                                          cl_int *                /* errcode_ret */);
    
    cl_int (CL_API_CALL *clSetMemObjectDestructorCallback)(  cl_mem /* memobj */,
                                                           void (CL_CALLBACK * /*pfn_notify*/)( cl_mem /* memobj */, void* /*user_data*/),
                                                           void * /*user_data */ );
    
    cl_mem (CL_API_CALL *clCreateSubBuffer)(cl_mem /* buffer */,
                                            cl_mem_flags /* flags */,
                                            cl_buffer_create_type /* buffer_create_type */,
                                            const void * /* buffer_create_info */,
                                            cl_int * /* error */);
											
    cl_int  (CL_API_CALL *clGetMemObjectInfo)(cl_mem /* memobj */,
                                              cl_mem_info /* param_name */,
                                              size_t /* param_value_size */,
                                              void * /* param_value */,
                                              size_t * /* param_value_size_ret */);

	cl_mem(CL_API_CALL *clCreateImage3D)(cl_context              /* context */,
											cl_mem_flags            /* flags */,
											const cl_image_format * /* image_format */,
											size_t                  /* image_width */,
											size_t                  /* image_height */,
											size_t                  /* image_depth */,
											size_t                  /* image_row_pitch */,
											size_t                  /* image_slice_pitch */,
											void *                  /* host_ptr */,
											cl_int *                /* errcode_ret */);
};


class DebayerOpenCLJob
{
private:
    unsigned int reserved1;
    unsigned int reserved2;

    DebayerOpenCLJob();
    ~DebayerOpenCLJob();
public:
    // returns the size in bytes of the result frame that the debayer job passed in will produce.
    static size_t ResultFrameSize( const DebayerOpenCLJob &debayerJob );

    // This will ensure all objects used for the frame are disposed of.
    // This call will block until the debayer on the queue executes, if the debayer has already executed no blocking will occur
    // This function must be called at a point after passing the debayerOpenCLJob into REDCL::processAsync
    // This will automatically be called upon destruction the DebayerOpenCLJob if it is not manually called earlier
    void completeAsync();

    
	// NOTE: This pointer + cl_mem & their contents must remain valid until completeAsync() or process() returns!
    void *raw_host_mem; 	// sized from r3dsdk buffer
    cl_mem raw_device_mem;  // sized from r3dsdk buffer

    size_t output_device_mem_size; 	// size of the output buffer
                                    // it must be a minimum of ResultFrameSize(job)

    cl_mem output_device_mem; // result/output buffer on the GPU
                              // if output_device_mem_size  must point to a cl_mem
                              //    object of the size output_device_mem_size
                              //    for the case of a cl_image this must be atleast the size of an
                              //    output image of the pixelType specified for processing
    
    VideoDecodeMode mode;

	// Image Processing Settings. This cannot be NULL.
    ImageProcessingSettings *imageProcessingSettings;

    // Supported Types are:
    //  PixelType_16Bit_RGB_Interleaved 
    //  PixelType_HalfFloat_RGB_Interleaved
    //  PixelType_HalfFloat_RGB_ACES_Int
    //
    // The user can also optionally use a cl_image object to gain more control over thier output pixel format. See the CLIMAGE_OUTPUT preprocessor define in the decode sample for more details.
    VideoPixelType pixelType;

private:
    void *reserved;
};

class REDCL
{
public:
    enum Status
    {
        Status_Ok = 0,
        Status_ErrorProcessing = 1,
        Status_InvalidJobParameter = 2,
        
        // mode value passed in is not compatible with this SDK or the mode used with the DecodeForGpuSdk call is not compatible
        Status_InvalidJobParameter_mode = 3,//mode is not compatible with this SDK or the mode used with the DecodeForGpuSdk call
        
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


	DebayerOpenCLJob *createDebayerJob();

    void releaseDebayerJob(DebayerOpenCLJob *job);

    // opencl_api_functions must be bound to your desired implementation of OpenCL.
    // compiledKernelCacheFolder is a folder path to where you want us to store and
	// load our compiled kernel files on disk. If you provide "" caching of compiled
	// kernels will be disabled which is NOT recommended as compile times can be long!
	// (see "OpenCL kernel caching.txt" for more information)
    REDCL(EXT_OCLAPI_1_1 &opencl_api_functions, const char *compiledKernelCacheFolder);

    ~REDCL();

    // checks to see if a context and command queue are compatible with REDCL
    // This call may take several seconds on slower cards as kernels get compiled
	// Decode kernels are compiled asynchronously. All OpenCL processing will be blocked
	// until that is complete. See "OpenCL kernel caching.txt" for more information
    Status checkCompatibility(cl_context context, cl_command_queue queue, cl_int &clerror);
    
    // processes a frame.
    // Thread safe blocking call
    // This function will enqueue the opencl commands and wait for the processing to complete
    //  	to ensure memory buffers are not released prior to completion of kernel
    Status process( cl_context context, cl_command_queue queue, DebayerOpenCLJob *debayerJob, cl_int &clerror);

    // enqueues all red processing on the current command queue
    // once processAsync has been called you will need to call debayerJob->completeAsync
    //      This will wait for commands enqueued by this sdk for the current frame to complete and then releases any additional resources.
    //      If the commands have already completed there is no wait involved
    Status processAsync( cl_context context, cl_command_queue queue, DebayerOpenCLJob *debayerJob, cl_int &clerror);

private:
    void *reserved;
};
}

#endif

