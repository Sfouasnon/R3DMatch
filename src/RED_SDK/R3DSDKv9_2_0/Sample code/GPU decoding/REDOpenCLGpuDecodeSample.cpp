/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#include <CoreServices/CoreServices.h>
#else
#include <CL/opencl.h>
#endif

#include <R3DSDK.h>
#include <R3DSDKOpenCL.h>

//To set the sample to use a cl_image instead of a buffer object
//enable the CL_IMAGE_OUTPUT define
//#define CLIMAGE_OUTPUT

namespace
{
size_t resultSize(size_t source, R3DSDK::VideoDecodeMode mode)
{
    if( mode == R3DSDK::DECODE_FULL_RES_PREMIUM )
    {
        return source;
    }
    
    if( mode == R3DSDK::DECODE_HALF_RES_GOOD ||  mode == R3DSDK::DECODE_HALF_RES_PREMIUM)
    {
        return source/2;
    }
    
    if( mode == R3DSDK::DECODE_QUARTER_RES_GOOD )
    {
        return source/4;
    }
    
    if( mode == R3DSDK::DECODE_EIGHT_RES_GOOD )
    {
        return source/8;
    }
    
    if( mode == R3DSDK::DECODE_SIXTEENTH_RES_GOOD )
    {
        return source/16;
    }
    return source;
}

//CL_API_CALL is an opencl definition it should be defined by the opencl implementation you are using in win32 it resolves to __stdcall
//				However this is left upto the opencl library that you are compiling against.

//In this sample we override clCreateBuffer and clReleaseBuffer so we can track the opencl buffer allocations and releases.
cl_mem CL_API_CALL local_clCreateBuffer(cl_context    context ,
                                cl_mem_flags  flags ,
                                size_t        size ,
                                void *        host_ptr ,
                                cl_int *      errcode_ret )
{
	cl_mem  memobj = clCreateBuffer(context, flags,size, host_ptr, errcode_ret);
	return memobj;
}
    
cl_int  CL_API_CALL local_clReleaseMemObject(cl_mem memobj )
{
    return clReleaseMemObject(memobj);
}

bool isIntelOpenCL(std::string vendor_name, std::string device_name)
{
    //Allow Iris cards to be used, but dissallow any other intel cards - ensuring that older Intel HD cards and integrated graphics are not chosen
    if(device_name.find("Iris") != std::string::npos)
        return true;
    
    if(vendor_name.find("Intel") != std::string::npos)
        return true;
    if(device_name.find("HD Graphics") != std::string::npos)
        return true;
    if(device_name.find("Intel") != std::string::npos)
        return true;
    
    
    return false;
}

//This SDK only supports OpenCL 1.1 and greater
//This function is used to identify if the platform only supports 1.0
bool isOpenCLPlatform1_0(cl_platform_id platform_id)
{
    char buffer[256];
    clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, 256, buffer, NULL);
    if (strstr(buffer, "OpenCL 1.0")) return true;

    return false;
}
   
//This SDK only supports OpenCL 1.1 and greater
//This function is used to identify if the device only supports 1.0 
bool isOpenCLDevice1_0( cl_device_id deviceId)
{
    char buffer[256];
    
#ifdef __APPLE__
    SInt32 versMaj, versMin, versBugFix;
    Gestalt(gestaltSystemVersionMajor, &versMaj);
    Gestalt(gestaltSystemVersionMinor, &versMin);
    Gestalt(gestaltSystemVersionBugFix, &versBugFix);
    
    //work around to ensure Devices on macOS 10.7.4 where CL_DRIVER_VERSION is used improperly are not used.
    if(versMaj == 10 && versMin == 7 && versBugFix == 4)
    {
        clGetDeviceInfo(deviceId, CL_DRIVER_VERSION, 256, buffer, NULL);
        if(strcmp(buffer,"1.0") == 0)
        {
            return true;
        }
    }
#endif
    clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, 256, buffer, NULL);
    if (strstr(buffer, "OpenCL 1.0")) return true;
    
    clGetDeviceInfo(deviceId, CL_DEVICE_OPENCL_C_VERSION, 256, buffer, NULL);
    if (strstr(buffer, "OpenCL 1.0")) return true;
    
    return false;
}

//Flag used for detecting when the decode is complete.
volatile bool decodeDone = false;
//Callback that the R3DSDK will call when the DecodeForGpuSdk task has completed.
void asyncCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus)
{
	printf( "Frame callback: %d\n", decodeStatus);
	*((R3DSDK::DecodeStatus*)item->PrivateData) = decodeStatus;
	decodeDone = true;
}

//This routine takes the output data from the R3DSDK::GpuDecoder::DecodeForGpuSdk and decompresses & debayers it into a usable image.
//This requires you to setup an OpenCL context, command queue and buffer for input and output.
//Then create a REDCL Context then a DebayerOpenCLJob
R3DSDK::REDCL::Status Debayer(void *source_raw_host_memory_buffer, size_t raw_buffer_size, R3DSDK::VideoPixelType pixelType, R3DSDK::VideoDecodeMode mode, size_t width, size_t height, R3DSDK::ImageProcessingSettings &ips, void **result_host_memory_buffer, size_t &result_buffer_size)
{
	//SETUP YOUR OPENCL API FUNCTION POINTERS
	//This is left for the SDK user to do, just incase you wish to use your own custom OpenCL dll
	R3DSDK::EXT_OCLAPI_1_1 api;

	api.clSetKernelArg = ::clSetKernelArg;
    api.clFlush = ::clFlush;
    api.clFinish = ::clFinish;
    api.clEnqueueCopyImage = ::clEnqueueCopyImage;

    api.clCreateContext = ::clCreateContext;
    api.clCreateCommandQueue = ::clCreateCommandQueue;
    api.clCreateSampler = ::clCreateSampler;
    api.clCreateKernel = ::clCreateKernel;
    api.clCreateBuffer = local_clCreateBuffer;
    api.clCreateProgramWithSource = ::clCreateProgramWithSource;
    api.clCreateProgramWithBinary = ::clCreateProgramWithBinary;
    api.clReleaseEvent = ::clReleaseEvent;
    api.clReleaseSampler = ::clReleaseSampler;
    api.clReleaseKernel = ::clReleaseKernel;
    api.clReleaseMemObject = local_clReleaseMemObject;
    api.clReleaseProgram = ::clReleaseProgram;
    api.clReleaseContext = ::clReleaseContext;
    api.clReleaseCommandQueue = ::clReleaseCommandQueue;
    api.clGetPlatformInfo = ::clGetPlatformInfo;
    api.clGetDeviceIDs = ::clGetDeviceIDs;
    api.clGetPlatformIDs = ::clGetPlatformIDs;
    api.clGetDeviceInfo = ::clGetDeviceInfo;
	api.clGetContextInfo = ::clGetContextInfo;
    api.clGetImageInfo =  ::clGetImageInfo;
    api.clGetProgramBuildInfo = ::clGetProgramBuildInfo;
    api.clGetProgramInfo = ::clGetProgramInfo;
    api.clGetKernelWorkGroupInfo = ::clGetKernelWorkGroupInfo;
    api.clBuildProgram = ::clBuildProgram;
    api.clEnqueueWriteBuffer = ::clEnqueueWriteBuffer;
    api.clEnqueueReadBuffer = ::clEnqueueReadBuffer;
    api.clEnqueueCopyBuffer = ::clEnqueueCopyBuffer;
    api.clEnqueueCopyBufferToImage = ::clEnqueueCopyBufferToImage;
    api.clEnqueueWriteImage = ::clEnqueueWriteImage;
    api.clEnqueueNDRangeKernel = ::clEnqueueNDRangeKernel;
    api.clEnqueueMapBuffer = ::clEnqueueMapBuffer;
    api.clEnqueueUnmapMemObject = ::clEnqueueUnmapMemObject;
    api.clWaitForEvents = ::clWaitForEvents;
    api.clEnqueueBarrier = ::clEnqueueBarrier;
    api.clEnqueueMarker = ::clEnqueueMarker;
    api.clGetMemObjectInfo = ::clGetMemObjectInfo;

	//OCL 1.1
	api.clCreateImage2D = ::clCreateImage2D;
	api.clCreateImage3D = ::clCreateImage3D;
    api.clSetMemObjectDestructorCallback = ::clSetMemObjectDestructorCallback;

    api.clCreateSubBuffer = ::clCreateSubBuffer;

	//Create the REDCL class
    //API is required but if compiledKernelCacheFolder parameter is
	//empty caching of kernels will be disabled which is not advisable!
	//(see "OpenCL kernel caching.txt" for more information)
	R3DSDK::REDCL *redcl = new R3DSDK::REDCL(api, "");

	//get platforms.
	cl_uint platforms_n = 0;
    int err = 0;
    try
    {
        err = clGetPlatformIDs(0, (cl_platform_id *)NULL, &platforms_n);
        if (err != CL_SUCCESS)
        {
            printf("Error: Failed to get a CL platform!\n");
			return R3DSDK::REDCL::Status_UnableToUseGPUDevice;
        }
    }
    catch(...)
    {
        printf( "Error: The OpenCL driver on this machine is not working! \n");
        return R3DSDK::REDCL::Status_UnableToUseGPUDevice;
    }

    cl_platform_id *platforms = new cl_platform_id [platforms_n];
    err = clGetPlatformIDs(platforms_n, platforms, NULL);
    if (err != CL_SUCCESS)
    {
        printf( "Error: Failed to get a CL platform IDs!\n");
        return R3DSDK::REDCL::Status_UnableToUseGPUDevice;
    }

	cl_platform_id platform_id;
	cl_device_id device_id;
	bool foundDevice = false;
	//get device
	for (cl_uint i = 0; i < platforms_n && !foundDevice; ++i)
    {
		cl_uint devices_n = 0;
		devices_n = 0;
        if (isOpenCLPlatform1_0(platforms[i]))
            continue;//We only support opencl 1.1 and higher, however this platform only supports OpenCL 1.0

		//We only support GPU devices. - NO CPU DEVICES!
        if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &devices_n) == CL_SUCCESS)
        {
            if (devices_n <= 0 )
			{//this platform has NO GPU Devices.
				continue;
			}

            cl_device_id *devices = new cl_device_id [devices_n];
            if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, devices_n, devices, NULL) == CL_SUCCESS)
            {
                for (cl_uint j=0; j < devices_n; ++j)
                {                      
                    size_t strSize=0;
                    
                    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, NULL, &strSize);
                    char *platform_name = new char [strSize+1];
                    clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, strSize, platform_name, NULL);
                    
                    strSize = 0;
                    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, NULL, &strSize);
                    char *platform_vendor = new char [strSize+1];
                    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, strSize, platform_vendor, NULL);
                    
                    if (isOpenCLDevice1_0(devices[j]))
					{	
						delete [] platform_vendor;
						delete [] platform_name;
                        continue;//We only support opencl 1.1 and higher, however this device only supports OpenCL 1.0
					}
                    
                    strSize = 0;
                    clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 0, NULL, &strSize);
                    char *device_name = new char [strSize+1];
                    clGetDeviceInfo(devices[j], CL_DEVICE_NAME, strSize, device_name, NULL);
                    
                    bool isOldIntelCard = isIntelOpenCL(platform_vendor, device_name );
                    
                    
                    delete [] device_name;
                    delete [] platform_vendor;
                    delete [] platform_name;

                    //Note: The intel check specifically skips IRIS cards as they are compatible.
                    //      So for Intel Iris cards this bool will be false allowing them to be used.
                    if( isOldIntelCard )
                        continue;

                    // Set platform id and name.
					platform_id = platforms[i];
					device_id = devices[j];
					foundDevice = true;
					break;
                }
            }
            delete [] devices;
        }
	}
	delete[] platforms;

	if(!foundDevice)
	{
		printf("Failed to find a capable device\n");
		return R3DSDK::REDCL::Status_UnableToUseGPUDevice;
	}


	//create OpenCL context
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform_id,
        0
    };
    //print final chosen card info
    {
        size_t strSize = 0;
        clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, 0, NULL, &strSize);
        char *platform_name = new char [strSize+1];
        clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, strSize, platform_name, NULL);
        printf("platform name: %s\n", platform_name);
        
        clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, 0, NULL, &strSize);
        char *platform_vendor = new char [strSize+1];
        clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, strSize, platform_vendor, NULL);
        printf("platform vendor: %s\n", platform_vendor);
        
        strSize = 0;
        clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, NULL, &strSize);
        char *device_name = new char [strSize+1];
        clGetDeviceInfo(device_id, CL_DEVICE_NAME, strSize, device_name, NULL);
        printf("device name: %s\n", device_name);
        
        
        delete [] device_name;
        delete [] platform_vendor;
        delete [] platform_name;
    }
    
    cl_context context = clCreateContext(contextProperties, 1, &device_id, NULL, NULL, &err);
    if (!context)
    {
       printf("Error: Failed to create an opencl context! Please check the system tab under preferences to ensure you have the correct hardware chosen. %d\n", err);
        return R3DSDK::REDCL::Status_UnableToUseGPUDevice;
    }
    if( err != CL_SUCCESS )
    {
        printf( "Error: Failed to create an opencl context! Please check the system tab under preferences to ensure you have the correct hardware chosen. %d\n", err);
        return R3DSDK::REDCL::Status_UnableToUseGPUDevice;
    }
    printf("Context using device: %p\n", (void *)device_id);

    cl_command_queue_properties queue_properties = {0};
	cl_command_queue queue = clCreateCommandQueue( context, device_id, queue_properties, &err);
	if( err != CL_SUCCESS )
    {
        printf( "Error: Failed to create command queue %d\n", err);
        return R3DSDK::REDCL::Status_UnableToUseGPUDevice;
    }
    
    R3DSDK::REDCL::Status compatible_status = redcl->checkCompatibility(context, queue, err);
    if( compatible_status != R3DSDK::REDCL::Status_Ok )
    {
        if( compatible_status == R3DSDK::REDCL::Status_UnableToLoadLibrary)
        {
            printf( "Error: Unable to load the REDCL dynamic library %d.\n", compatible_status);
            return compatible_status;
        }
        printf( "Error: Graphics card driver is not compatible %d\n", compatible_status);
        return R3DSDK::REDCL::Status_UnableToUseGPUDevice;
    }
    
	//setup your Debayer settings.
	R3DSDK::DebayerOpenCLJob *data = redcl->createDebayerJob();
	data->mode = mode; //Quality mode
    data->pixelType = pixelType;
	data->raw_host_mem = source_raw_host_memory_buffer;

    data->imageProcessingSettings = new R3DSDK::ImageProcessingSettings();
    *data->imageProcessingSettings = ips;

    //UPLOAD the buffer you got from R3DSDK::GpuDecoder::DecodeForGPUSDK to the Graphics card.
	//determine the needed buffer sizes.
	result_buffer_size = R3DSDK::DebayerOpenCLJob::ResultFrameSize( *data );

	//create raw buffer on the OpenCL device
	data->raw_device_mem = clCreateBuffer(context, CL_MEM_READ_ONLY, raw_buffer_size, NULL, &err );
	//upload the result from the GpuDecoder to OpenCL
	err = clEnqueueWriteBuffer(queue, data->raw_device_mem, CL_TRUE, 0, raw_buffer_size, data->raw_host_mem, 0,NULL,NULL);
	if( err != CL_SUCCESS )
	{
		printf("Failed to move raw frame to card %d\n", err);
		return R3DSDK::REDCL::Status_InvalidJobParameter_raw_device_mem;
	}

    //for the purpose of this sample we use primitive clFinish for synchronization.  
    //clFinish is not a good choice for real applications.
	err = clFinish(queue);
	if( err != CL_SUCCESS )
	{
		printf("Failed to finish after moving raw frame to card %d\n", err);
		return  R3DSDK::REDCL::Status_InvalidJobParameter_raw_device_mem;
	}

	

	//create the result image buffer in OpenCL
	//You must specify an existing buffer for the result image
	//Be sure to set DebayerOpenCLJob::output_device_mem_size >= result_buffer_size
	//and a pointer to the device buffer in DebayerOpenCLJob::output_device_mem
#ifdef CLIMAGE_OUTPUT
    //using CL_IMAGE output does not work with CL_UNSIGNED_INT{8:16:32} types
    // cl_image object is written to with write_imagef
    //  This method gives the SDK user more flexibility with pixeltypes as they can request an image using one SDK pixel type for processing and retrieve the data in the pixel format they specified in the cl_image instead.
    
    cl_image_format format = { CL_RGBA, CL_UNORM_INT16};
    //cl_image_format format = { CL_RGBA, CL_HALF_FLOAT };

    cl_image_desc desc = {CL_MEM_OBJECT_IMAGE2D, width, height, 0, 2, 0, 0, 0, 0, NULL};
    //Device output mem must be atleast the size of the pixeltype specified for processing (The R3DSDK::VideoPixelType)
    result_buffer_size = width * height * 4U * 2U; //(16Bit RGB)
    data->output_device_mem = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL,&err);
#else
	data->output_device_mem = clCreateBuffer(context, CL_MEM_READ_WRITE, result_buffer_size, NULL, &err );
#endif
	data->output_device_mem_size = result_buffer_size;
	if( err != CL_SUCCESS )
	{
		printf("Failed to allocate result frame on card %d\n", err);
		return R3DSDK::REDCL::Status_ErrorProcessing;
	}

	//Do decompression and run the debayer on the given OpenCL buffers.
	bool process_async = true;
	if( process_async )
	{
        R3DSDK::REDCL::Status status = redcl->processAsync( context, queue, data, err );
        printf( "Result: %d %p\n", status, data->output_device_mem);
		if( status != R3DSDK::REDCL::Status_Ok )
		{
			printf("Failed to process frame %d", status);
			if( err != CL_SUCCESS )
			{
				printf(" OpenCL Error: %d\n", err);
				return status;
			}
			printf("\n");
            return status;
		}
		
		//enqueue other opencl comamnds etc here to the queue.
		//you can do any opencl queue synchronization here you need.

		//This will ensure all internal objects used for the frame are disposed of.
		//This call will block until the debayer on the queue executes, 
		// if the debayer has already executed no block will occur
		data->completeAsync();
	}
	else
	{
        //Process simply condenses the processAsync call with completeAsync, effectively blocking until all SDK tasks for this frame have completed.
        //The downside to using process insead of processAsync is that you are stuck synchronizing before executing your own kernels (which if on the same queue do not need synchronization).
        //The benefit is that it simplifies the usage of the SDK.
		R3DSDK::REDCL::Status status = redcl->process( context, queue, data , err); 
		printf( "Result: %d %p\n", status, data->output_device_mem);
		if( status != R3DSDK::REDCL::Status_Ok )
		{
			printf("Failed to process frame %d", status);
			if( err != CL_SUCCESS )
			{
				printf(" OpenCL Error: %d\n", err);
				return status;
			}
			printf("\n");
            return status;
		}
	}


    //Optional: Copy the result frame back into CPU/Host memory to be written to disk later

	//allocate the result buffer in host memory.
	*result_host_memory_buffer = malloc(result_buffer_size);//RGB 16 Interleaved
	//read the OpenCL buffer back to the host memory result buffer. - Note this is not always the optimal way to read back. (Use pinned memory in a real app)
#ifdef CLIMAGE_OUTPUT
    size_t origin[3] = {0,0,0};
    size_t region[3] = {width,height,1};
    err = clEnqueueReadImage(queue, data->output_device_mem, CL_TRUE, origin, region, 0, 0, *result_host_memory_buffer, 0,NULL,NULL);
#else
	err = clEnqueueReadBuffer(queue, data->output_device_mem, CL_TRUE, 0, result_buffer_size, *result_host_memory_buffer, 0,NULL,NULL);
#endif
	if( err != CL_SUCCESS )
	{
		printf("Failed to read result frame from card %d\n", err);
		return R3DSDK::REDCL::Status_ErrorProcessing;
	}
	//ensure the read is complete.
	err = clFinish(queue);
	if( err != CL_SUCCESS )
	{
		printf("Failed to finish after reading result frame from card %d\n", err);
		return R3DSDK::REDCL::Status_ErrorProcessing;
	}

    //free memory objects on the data object
    err = clReleaseMemObject(data->output_device_mem);
    if( err != CL_SUCCESS )
    {
        printf("Failed to release memory object %d\n", err);
        return R3DSDK::REDCL::Status_ErrorProcessing;
    }

    err = clReleaseMemObject(data->raw_device_mem);
    if( err != CL_SUCCESS )
    {
        printf("Failed to release memory object %d\n", err);
        return R3DSDK::REDCL::Status_ErrorProcessing;
    }

	//final image data is now in result_buffer
    //release the job before deleting the context
    redcl->releaseDebayerJob(data);

	//tear down redCL
	delete redcl;

	//Tear down OpenCL
	//release other OpenCL objects. 
	err = clReleaseCommandQueue(queue);
	if( err != CL_SUCCESS )
	{
		printf("Failed to release command queue %d\n", err);
		return R3DSDK::REDCL::Status_ErrorProcessing;
	}

	err = clReleaseContext(context);
	if( err != CL_SUCCESS )
	{
		printf("Failed close the context %d\n", err);
		return R3DSDK::REDCL::Status_ErrorProcessing;
	}
	return R3DSDK::REDCL::Status_Ok;
}

// The R3D SDK requires that the output buffer is 16-byte aligned
// and the number of bytes per row is as well. The latter is always
// true in the current situation with the RED ONE resolutions, the
// 16-bit planar RGB the library is outputting and an aligned buffer
// sizeNeeded will be updated to indicate how many bytes were needed
// to properly align the buffer
unsigned char * AlignedMalloc(size_t & sizeNeeded)
{
	// alloc 15 bytes more to make sure we can align the buffer in case it isn't
	unsigned char * buffer = (unsigned char *)malloc(sizeNeeded + 15U);

	if (!buffer)
		return NULL;

	sizeNeeded = 0U;

	// cast to a 32-bit or 64-bit (depending on platform) integer so we can do the math
	uintptr_t ptr = (uintptr_t)buffer;

	// check if it's already aligned, if it is we're done
	if ((ptr % 16U) == 0U)
		return buffer;

	// calculate how many bytes we need
	sizeNeeded = 16U - (ptr % 16U);

	return buffer + sizeNeeded;
}

R3DSDK::DecodeStatus Decompress(const char *filename, size_t frame_number, R3DSDK::ImageProcessingSettings &ips_to_be_filled_with_defaults, R3DSDK::VideoDecodeMode mode, size_t &source_width, size_t &source_height, void **raw_buffer, size_t &raw_buffer_size, size_t &raw_buffer_aligned_ptr_adjustment)
{
	R3DSDK::Clip *clip = new R3DSDK::Clip(filename);
	if( clip->Status() != R3DSDK::LSClipLoaded )
	{
		printf("Failed to load clip %d\n", clip->Status());
		return R3DSDK::DSNoClipOpen;
	}
    
    source_width = clip->Width();
    source_height = clip->Height();

	//setup R3DSDK to decode a frame
	R3DSDK::AsyncDecompressJob *job = new R3DSDK::AsyncDecompressJob();
	job->Clip = clip;
	job->Mode = mode;

	raw_buffer_size = R3DSDK::GpuDecoder::GetSizeBufferNeeded( *job );
	raw_buffer_aligned_ptr_adjustment = raw_buffer_size;
	*raw_buffer = AlignedMalloc(raw_buffer_aligned_ptr_adjustment);

	job->OutputBuffer = *raw_buffer;
	job->OutputBufferSize = raw_buffer_size;
	job->PrivateData = new R3DSDK::DecodeStatus();
	*((R3DSDK::DecodeStatus*)job->PrivateData) = R3DSDK::DSOutputBufferInvalid;
	job->VideoFrameNo = 0;
	job->VideoTrackNo = 0;
	job->Callback = asyncCallback;

	R3DSDK::GpuDecoder *decoder = new R3DSDK::GpuDecoder();
	decoder->Open();
	//Load a frame
	R3DSDK::DecodeStatus decompress_status = decoder->DecodeForGpuSdk( *job ); 
	if( decompress_status != R3DSDK::DSDecodeOK )
	{
		printf("Failed to start decompression %d\n", decompress_status);
		//abort clearing up allocated memory.
		decoder->Close();
		delete (R3DSDK::DecodeStatus*)job->PrivateData;
		delete job;
		delete decoder;
		clip->Close();
		delete clip;
		// free the original pointer, not the one adjusted for alignment
		free(((unsigned char *)*raw_buffer) - raw_buffer_aligned_ptr_adjustment);
		return decompress_status;
	}
	while(!decodeDone )
	{
		//terribly evil spin lock until job completes.
	}
	R3DSDK::DecodeStatus callback_status = *((R3DSDK::DecodeStatus*)job->PrivateData);
	//tear down the GpuDecoder as we are done with that now
	decoder->Close();
	delete (R3DSDK::DecodeStatus*)job->PrivateData;
	delete job;
	delete decoder;

	//get the image processing settings for the frame from the clip.
	
    clip->GetDefaultImageProcessingSettings(ips_to_be_filled_with_defaults);
    
	//we no longer need the clip around.
	clip->Close();
	delete clip;
	
	return callback_status;
}//end Decompress

}


//Loads a frame using the R3DSDK::GpuDecoder API
//Then decompresses & debayers the frame using the R3DSDK::REDCL API
//Then writes the frame to disk.
int main(int argc, char **argv)
{
	if( argc < 2 )
	{
		printf("Invalid number of arguments\nExample: %s path_to_clip\n", argv[0]);
		return 4;
	}

    std::string error;

    R3DSDK::InitializeStatus init_status = R3DSDK::InitializeSdk(".", OPTION_RED_OPENCL);
    if( init_status != R3DSDK::ISInitializeOK )
    {
        R3DSDK::FinalizeSdk();
        printf("Failed to load R3DSDK, %d\n", init_status);
        return 4;
    }
    

    R3DSDK::Clip *clip = new R3DSDK::Clip(argv[1]);
    if( clip->Status() != R3DSDK::LSClipLoaded )
    {
        R3DSDK::FinalizeSdk();
        printf("Failed to load clip %d\n", clip->Status());
        return R3DSDK::DSNoClipOpen;
    }

	R3DSDK::VideoDecodeMode mode = R3DSDK::DECODE_HALF_RES_PREMIUM;
	
    

    R3DSDK::VideoPixelType pixelType = R3DSDK::PixelType_16Bit_RGB_Interleaved;
	R3DSDK::ImageProcessingSettings *ips = new R3DSDK::ImageProcessingSettings();
	void *raw_buffer = NULL;
	size_t raw_buffer_size = 0;
	size_t raw_buffer_aligned_ptr_adjustment = 0;
    size_t source_width = 0;
    size_t source_height = 0;
		
	R3DSDK::DecodeStatus decompress_status = Decompress(argv[1], 0, *ips, mode, source_width, source_height, &raw_buffer, raw_buffer_size, raw_buffer_aligned_ptr_adjustment );
	if( decompress_status != R3DSDK::DSDecodeOK )
	{
		//failed to decode
		printf("Error decompressing frame: %d\n", decompress_status );
        R3DSDK::FinalizeSdk();
		return 5;
	}

    size_t result_width = 0;
    size_t result_height = 0;
    result_width = resultSize(source_width, mode);
    result_height = resultSize(source_height, mode);
	//size the result buffer will be returned from debayer
	size_t result_buffer_size = 0;
	//buffer in host memory filled by example code in debayer routine.
	void *result_host_memory_buffer = NULL;

	//Debayer Logic has been completely seperated to illustrate it's independence from the decompression step.
	//Debayer a Raw Buffer into a frame of the selected pixel type
	R3DSDK::REDCL::Status debayer_status = Debayer( raw_buffer, raw_buffer_size, pixelType, mode, result_width, result_height, *ips, &result_host_memory_buffer, result_buffer_size );
	if( raw_buffer )
	{
		// free the original pointer, not the one adjusted for alignment
		free(((unsigned char *)raw_buffer) - raw_buffer_aligned_ptr_adjustment);
	}

	if( debayer_status == R3DSDK::REDCL::Status_Ok )
	{
		//Write the result image buffer to an output File
		FILE *f = fopen("out.raw", "wb");
		if( f )
		{
            fwrite(result_host_memory_buffer, 1, result_buffer_size, f);
            fflush(f);
            fclose(f);
            printf("file saved.\n");
		}
		else
		{
			printf("failed to create output file.\n");
		}
	}
	else 
	{
		//failed to debayer
		printf("Error debayering frame: %d\n", debayer_status );
		//getchar();
        R3DSDK::FinalizeSdk();
		return 6;
	}
	
	if( result_host_memory_buffer )
		free(result_host_memory_buffer);
    delete ips;
    
    R3DSDK::FinalizeSdk();
	return 0;
}
