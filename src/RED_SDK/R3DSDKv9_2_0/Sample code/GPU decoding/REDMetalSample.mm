/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

#include <Metal/Metal.h>
#include "R3DSDK.h"
#include "R3DSDKMetal.h"
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string.h>

namespace
{
    volatile bool decodeDone = false;
    void asyncCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus)
    {
        printf( "Frame callback: %d\n", decodeStatus);
        *((R3DSDK::DecodeStatus*)item->PrivateData) = decodeStatus;
        decodeDone = true;
    }
}

namespace
{
    //This routine takes the output data from the R3DSDK::AsyncDecoder::DecodeForGpuSdk and debayers it into a usable image.
    //This requires you to setup the Metal environment, streams and buffer for input and output.
    //Then create a REDMetal Context then a DebayerMetalJob
    R3DSDK::REDMetal::Status Debayer(void *source_raw_host_memory_buffer, size_t raw_buffer_size, R3DSDK::VideoPixelType pixelType, R3DSDK::VideoDecodeMode mode, R3DSDK::ImageProcessingSettings &ips, void **result_host_memory_buffer, size_t &result_buffer_size)
    {
        @autoreleasepool {
            
        // open default device
        // - this will be the high performance one (NVIDIA/AMD) instead of the low power (INTEL, see lowPower property)
        // - on Mac Pro this will return the one with the display(s) attached
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        
        if (device == nil)
            return R3DSDK::REDMetal::Status_UnableToUseGPUDevice;
        
        BOOL supportsOSXGpu1_1 = [device supportsFeatureSet:MTLFeatureSet_OSX_GPUFamily1_v1];

        if (!supportsOSXGpu1_1)
            return R3DSDK::REDMetal::Status_UnableToUseGPUDevice;
        
        id<MTLCommandQueue> queue = [device newCommandQueue];
        
        //create the REDMetal class
        R3DSDK::EXT_METAL_API api;	// default instance is okay if no memory management is desired
        R3DSDK::REDMetal *redmetal = new R3DSDK::REDMetal(api);
        int err = 0;
        
        R3DSDK::REDMetal::Status status = redmetal->checkCompatibility( queue, err );
        
        if( R3DSDK::REDMetal::Status_Ok != status )
        {
            printf("Compatibility Check Failed\n");
            return R3DSDK::REDMetal::Status_UnableToUseGPUDevice;
        }
        
        //allocate the debayer job
        R3DSDK::DebayerMetalJob *data = redmetal->createDebayerJob();
        
        data->imageProcessingSettings = new R3DSDK::ImageProcessingSettings();
        data->mode = mode; //Quality mode
        
        data->raw_host_mem = source_raw_host_memory_buffer;
        memcpy(data->imageProcessingSettings, &ips, sizeof(R3DSDK::ImageProcessingSettings));//Image Processing Settings to apply
        data->pixelType = pixelType;
        //verify buffer size
        if( raw_buffer_size == 0 )
        {
            return R3DSDK::REDMetal::Status_InvalidJobParameter_raw_host_mem;
        }
        
        //create raw buffer on the Metal device
        data->raw_device_mem = [device newBufferWithBytes:source_raw_host_memory_buffer length:raw_buffer_size options:MTLResourceStorageModeManaged];
            
        if( data->raw_device_mem == nil )
        {
            printf("Failed to move raw frame to card\n");
            return R3DSDK::REDMetal::Status_ErrorProcessing;
        }
        
        //setup result pointer - will be a device memory pointer
        result_buffer_size = R3DSDK::DebayerMetalJob::ResultFrameSize(*data);
            
            if (result_buffer_size == 0)
            {
                printf("ResultFrameSize returned 0\n");
                return R3DSDK::REDMetal::Status_ErrorProcessing;
            }

        //YOU MUST specify an existing buffer for the result image
        //Set DebayerMetalJob::output_device_mem_size >= result_buffer_size
        //and a pointer to the device buffer in DebayerMetalJob::output_device_mem
        data->output_device_mem = [device newBufferWithLength:result_buffer_size options:MTLResourceStorageModeManaged];
        data->output_device_mem_size = result_buffer_size;
        
        if( data->output_device_mem == nil )
        {
            printf("Failed to allocate result frame on card\n");
            return R3DSDK::REDMetal::Status_ErrorProcessing;
        }
            
        bool process_debayer_async = true;
        //Run the debayer on the given buffers.
        if( process_debayer_async )
        {
            R3DSDK::REDMetal::Status status = redmetal->processAsync( queue, data, err );
            
            if( status != R3DSDK::REDMetal::Status_Ok )
            {
                printf("Failed to process frame %d", status);   // err is currently unused
                printf( "\n");
                return status;
            }
            
            
            
            //enqueue other Metal comamnds etc here to the stream.
            //you can do any Metal stream synchronization here you need.
            
            //This will ensure all objects used for the frame are disposed of.
            //This call will block until the debayer on the stream executes,
            // if the debayer has already executed no block will occur
            data->completeAsync();
        }
        else
        {
            //Process simply condenses the processAsync call with completeAsync, effectively blocking until all SDK tasks for this frame have completed.
            //The downside to using process insead of processAsync is that you are stuck synchronizing before executing your own kernels.
            //The benefit is that it simplifies the usage of the SDK.
            R3DSDK::REDMetal::Status status = redmetal->process( queue, data, err );
            
            if( status != R3DSDK::REDMetal::Status_Ok )
            {
                printf("Failed to process frame %d", status);   // err is currently unused
                printf( "\n");
                return status;
            }
        }
        
        //Optional: Copy the image back to cpu/host memory to be written to disk later
        
        //allocate the result buffer in host memory.
        *result_host_memory_buffer = malloc(result_buffer_size);//RGB 16 Interleaved
        
        
        //read the GPU buffer back to the host memory result buffer.
        id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
        [blitEncoder synchronizeResource:data->output_device_mem];
        [blitEncoder endEncoding];
        [commandBuffer commit];
        
        //ensure the read is complete.
        [commandBuffer waitUntilCompleted];

        memcpy( *result_host_memory_buffer,  [data->output_device_mem contents], result_buffer_size );
        
        //release non obj-C memory
        delete data->imageProcessingSettings;
        data->imageProcessingSettings =  NULL;
        
        redmetal->releaseDebayerJob(data);

        //tear down redMetal
        delete redmetal;
        
        return R3DSDK::REDMetal::Status_Ok;
        
        }//end @autoreleasepool
    }//end Debayer
    
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
    
    R3DSDK::DecodeStatus Decompress(const char *filename, size_t frame_number, R3DSDK::ImageProcessingSettings &ips_to_be_filled_with_defaults, R3DSDK::VideoDecodeMode mode, void **raw_buffer, size_t &raw_buffer_size, size_t &raw_buffer_aligned_ptr_adjustment)
    {
        R3DSDK::Clip *clip = new R3DSDK::Clip(filename);
        if( clip->Status() != R3DSDK::LSClipLoaded )
        {
            printf("Failed to load clip %d\n", clip->Status());
            return R3DSDK::DSNoClipOpen;
        }
        
        printf("Clip res = %u x %u\n", (unsigned int)clip->Width(), (unsigned int)clip->Height());
        
        //setup R3DSDK to decode a frame
        R3DSDK::AsyncDecompressJob *job = new R3DSDK::AsyncDecompressJob();
        job->Clip = clip;
        job->Mode = mode;
        
        
        raw_buffer_size = R3DSDK::AsyncDecoder::GetSizeBufferNeeded( *job );
        raw_buffer_aligned_ptr_adjustment = raw_buffer_size;
        *raw_buffer = AlignedMalloc(raw_buffer_aligned_ptr_adjustment);
        
        job->OutputBuffer = *raw_buffer;
        job->OutputBufferSize = raw_buffer_size;
        job->PrivateData = new R3DSDK::DecodeStatus();
        *((R3DSDK::DecodeStatus*)job->PrivateData) = R3DSDK::DSOutputBufferInvalid;
        job->VideoFrameNo = frame_number;
        job->VideoTrackNo = 0;
        job->Callback = asyncCallback;
        
        R3DSDK::AsyncDecoder *r3dAsync = new R3DSDK::AsyncDecoder();
        r3dAsync->Open(r3dAsync->ThreadsAvailable(), false);
        //Decode a frame using the R3DSDK
        R3DSDK::DecodeStatus decompress_status = r3dAsync->DecodeForGpuSdk( *job );
        if( decompress_status != R3DSDK::DSDecodeOK )
        {
            printf("Failed to start decompression %d\n", decompress_status);
            
            //abort clearing up allocated memory.
            r3dAsync->Close();
            delete (R3DSDK::DecodeStatus*)job->PrivateData;
            delete job;
            delete r3dAsync;
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
        //tear down the Async decoder as we are done with that now
        r3dAsync->Close();
        delete (R3DSDK::DecodeStatus*)job->PrivateData;
        delete job;
        delete r3dAsync;
        
        //get the image processing settings for the frame from the clip.
        
        clip->GetDefaultImageProcessingSettings(ips_to_be_filled_with_defaults);
        
        //we no longer need the clip around.
        clip->Close();
        delete clip;
        
        return callback_status;
    }//end Decompress
    
}//end anonymous namespace

//Decompresses a frame using the R3DSDK::AsyncDecoder API
//Then Debayers the frame using the R3DSDK::REDMetal API
//Then writes the frame to disk.
int main(int argc, char **argv)
{
    if( argc < 2 )
    {
        printf("Invalid number of arguments\nExample: %s path_to_clip\n", argv[0]);
        return 4;
    }
    
    R3DSDK::InitializeStatus init_status = R3DSDK::InitializeSdk(".", OPTION_RED_METAL);
    if( init_status != R3DSDK::ISInitializeOK )
    {
        R3DSDK::FinalizeSdk();
        printf("Failed to load R3DSDK Lib: %d\n", init_status);
        return 42;
    }
        
    R3DSDK::VideoDecodeMode mode = R3DSDK::DECODE_HALF_RES_PREMIUM;
    R3DSDK::VideoPixelType pixelType = R3DSDK::PixelType_16Bit_RGB_Interleaved;
    R3DSDK::ImageProcessingSettings *ips = new R3DSDK::ImageProcessingSettings();
    void *raw_buffer = NULL;
    size_t raw_buffer_size = 0;
    size_t raw_buffer_aligned_ptr_adjustment = 0;
    
    //Decompress the R3D Source frame into a Raw Buffer using the AsnycDecompressor
    R3DSDK::DecodeStatus decompress_status = Decompress(argv[1], 0, *ips, mode , &raw_buffer, raw_buffer_size, raw_buffer_aligned_ptr_adjustment );
    if( decompress_status != R3DSDK::DSDecodeOK )
    {
        //failed to decode
        printf("Error decompressing frame: %d\n", decompress_status );
        R3DSDK::FinalizeSdk();
        return 5;
    }
    
    //size the result buffer will be returned from debayer
    size_t result_buffer_size = 0;
    //buffer in host memory filled by example code in debayer routine.
    void *result_host_memory_buffer = NULL;
    
    //Debayer Logic has been completely seperated to illustrate it's independence from the decompression step.
    //Debayer a Raw Buffer into a frame of the selected pixel type
    R3DSDK::REDMetal::Status debayer_status = Debayer( raw_buffer, raw_buffer_size, pixelType, mode, *ips, &result_host_memory_buffer, result_buffer_size );
    if( raw_buffer )
    {
        // free the original pointer, not the one adjusted for alignment
        free(((unsigned char *)raw_buffer) - raw_buffer_aligned_ptr_adjustment);
    }
    
    if( debayer_status == R3DSDK::REDMetal::Status_Ok )
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
        R3DSDK::FinalizeSdk();
        return 6;
    }
    
    if( result_host_memory_buffer )
        free(result_host_memory_buffer);
    delete ips;
    
    R3DSDK::FinalizeSdk();
    return 0;
}
