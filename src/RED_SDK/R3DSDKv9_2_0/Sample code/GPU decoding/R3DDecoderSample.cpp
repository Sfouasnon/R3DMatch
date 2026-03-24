/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

#include <R3DSDK.h>

#include <R3DSDKDecoder.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
using namespace R3DSDK;

namespace
{

volatile size_t decodeDone = 0;
void asyncCallback(R3DSDK::R3DDecodeJob * item, R3DSDK::R3DStatus decodeStatus)
{
	item->privateData = NULL;//set the job as not in use (Sample app specific usage)
	++decodeDone;
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
	unsigned char * buffer = (unsigned char *)malloc(sizeNeeded + 1023U);

	if (!buffer)
		return NULL;

	sizeNeeded = 0U;

	// cast to a 32-bit or 64-bit (depending on platform) integer so we can do the math
	uintptr_t ptr = (uintptr_t)buffer;

	// check if it's already aligned, if it is we're done
	if ((ptr % 1024U) == 0U)
		return buffer;

	// calculate how many bytes we need
	sizeNeeded = 1024U - (ptr % 1024U);

	return buffer + sizeNeeded;
}
    
enum DecoderType
{
DecoderType_GPU_Cuda,
DecoderType_GPU_OCL
};

//Helper routine which when you specify a decoder type it passes back an instance of R3DDecoder using every available device.
R3DSDK::R3DStatus setup_R3DDecoder(R3DDecoder **decoder_out, DecoderType decoder_type = DecoderType_GPU_OCL)
{

	R3DSDK::R3DDecoderOptions *options = NULL;
	R3DStatus status = R3DSDK::R3DDecoderOptions::CreateOptions(&options);
	options->setMemoryPoolSize( 1024U );
	options->setGPUMemoryPoolSize( 1024U );
	options->setGPUConcurrentFrameCount( 1U );
	options->setScratchFolder("");//empty string disables scratch folder
	options->setDecompressionThreadCount(0U);//cores - 1 is good if you are a gui based app.
	options->setConcurrentImageCount(0U);//threads to process images/manage state of image processing.

	if( decoder_type == DecoderType_GPU_Cuda )
	{
        std::vector<CudaDeviceInfo > cudaDevices;
        status = options->GetCudaDeviceList(cudaDevices);
        if( status != R3DSDK::R3DStatus_Ok )
        {
            return status;
        }
		
		if( cudaDevices.size() == 0 )
		{
			return R3DSDK::R3DStatus_NoGPUDeviceSpecified;
		}

		for(std::vector<CudaDeviceInfo>::iterator it = cudaDevices.begin(); it != cudaDevices.end(); ++it)
		{
			options->useDevice( *it );
		}
	}
	else if(decoder_type == DecoderType_GPU_OCL)
	{
        std::vector<OpenCLDeviceInfo > openclDevices;
		status = options->GetOpenCLDeviceList(openclDevices);
        if( status != R3DSDK::R3DStatus_Ok )
        {
            return status;
        }
		if( openclDevices.size() == 0 )
		{
			return R3DSDK::R3DStatus_NoGPUDeviceSpecified;
		}
		for(std::vector<OpenCLDeviceInfo>::iterator it = openclDevices.begin(); it != openclDevices.end(); ++it)
		{
			options->useDevice( *it );
		}
	}

	status = R3DSDK::R3DDecoder::CreateDecoder(options, decoder_out);
	
	R3DSDK::R3DDecoderOptions::ReleaseOptions(options);
	return status;
}
}

int main(int argc, char **argv)
{
    using namespace R3DSDK;
	if( argc < 2 )
	{
		printf("Invalid number of arguments\nExample: %s path_to_clip\n", argv[0]);
		return -5;
	}

    R3DSDK::InitializeStatus init_status = R3DSDK::InitializeSdk(".", OPTION_RED_DECODER);
    if( init_status != R3DSDK::ISInitializeOK )
    {
        FinalizeSdk();
    	printf("Unable to load R3D Dynamic lib %d\n", init_status);
    	return -194;
    }
    printf("SDK Initialized\n");
	VideoDecodeMode mode = DECODE_HALF_RES_PREMIUM;
	VideoPixelType pixelType = PixelType_16Bit_RGB_Interleaved;

	//create and R3DSDK Clip from path
	Clip *clip = new Clip(argv[1]);
	if( clip->Status() != LSClipLoaded )
	{
		printf("Failed to load clip %d\n", clip->Status());
        FinalizeSdk();
		return -3;
	}
	
	printf("Creating decoder\n");
	R3DDecoder *decoder = NULL;
	//Create an R3DDecoder instance
	R3DStatus status = setup_R3DDecoder(&decoder, DecoderType_GPU_OCL);
	if( status != R3DStatus_Ok )
	{
        if( status == R3DStatus_UnableToLoadLibrary)
        {
            printf( "Error: Unable to load the R3DDecoder dynamic library %d, This could be caused by the file being missing, or potentially missing the cudart or OpenCL dynamic library.\n", status);
	        FinalizeSdk();
            return status;
        }

        
		printf("Unable to create R3DDecoder instance Error: %d\n", status);
        FinalizeSdk();
		return -1;
	}

	size_t count = 1000;
	printf("Decoding frame\n");

    // setup a list of jobs
    //the number of jobs to be used determines how many output buffers are allocated at a single time.
    std::vector<R3DSDK::R3DDecodeJob*> jobs;
    std::vector<size_t> alignedSizes;
    size_t simultaneousJobs = 16;
    for( int i = 0; i < simultaneousJobs; ++i)
    {
		//create a decode job
		R3DSDK::R3DDecodeJob *job = NULL;
		R3DDecoder::CreateDecodeJob(&job);
        size_t frame_number = simultaneousJobs % clip->VideoFrameCount();
        job->clip = clip;
        job->mode = mode;
        job->pixelType = pixelType;
        
		//output buffer sized for 16bit RGB image size
        size_t outputBufferSize = clip->Width()  * clip->Height() * 3U * 2U;
        size_t alignedSize = outputBufferSize;
		//create the aligned output buffer
        void *outputBuffer = AlignedMalloc(alignedSize);
		//setup bytes per row for 16bit RGB
        job->bytesPerRow = 3U * 2U * clip->Width();
        job->outputBuffer = outputBuffer;
        job->outputBufferSize = outputBufferSize;
        //this sample app uses privateData to store if the Job is in use or not.
        job->privateData = NULL;
        job->videoFrameNo = frame_number;
        job->videoTrackNo = 0;
        job->imageProcessingSettings = new R3DSDK::ImageProcessingSettings();
		clip->GetDefaultImageProcessingSettings(*(job->imageProcessingSettings));
        job->callback = asyncCallback;
        jobs.push_back(job);

		//save aligned size for proper deletion of aligned buffer ptr at end.
        alignedSizes.push_back(alignedSize);
    }
    
    size_t submittedCount = 0;
	//keep looping until all frames are submitted.
	//this loop is limited by the availability of jobs.
	// as decodes complete and jobs are freed they will be reused until we reach the desired frame count submitted
    while( submittedCount < count )
    {
        R3DSDK::R3DDecodeJob *nextJob = NULL;
        while(!nextJob)
        {
            //a proper application would use a mutex or wait condition to determine when a job is available for usage.
            //however this is just a sample app, to keep this simple and the dependency count down we'll just spin until one is available.
            for(int jobIdx = 0; jobIdx < simultaneousJobs; ++jobIdx)
            {//for all jobs check to see if one is available
                if( jobs[jobIdx]->privateData == NULL )
                {
                    nextJob = jobs[jobIdx];
                    break;//break out of the loop once we find an available job.
                }
            }
        }
        //we should have a job to decode with at this point.
		//mark the job as in use.
        nextJob->privateData = (void*)-1;
		//here we could update any paramters on the job if we wanted to do a different frame or such.
		//this particular sample however just keeps decoding the same frames with the same settings initially setup.

		//start the decode of the job.
		R3DStatus status = decoder->decode(nextJob);
        if( status != R3DStatus_Ok )
		{
			printf("Error starting decode: %d\n", status);
	        FinalizeSdk();
			return -2;
		}
		//we successfully started processing of another frame, increment the submit count
        submittedCount++;
    }
    
	printf("Waiting for frames to complete\n");
	
	while(decodeDone != count)
	{
		//all frames have been submitted but not all frames have completed yet, waiting for all frames to finish prior to ending the application
	}

	//all frames have completed decoding
	printf("Decode complete\n");
    
    //cleanup buffers
    for( int i = 0; i < simultaneousJobs; ++i)
    {
        free(((char *)jobs[i]->outputBuffer) - alignedSizes[i]);
        if(jobs[i]->imageProcessingSettings)
        	delete jobs[i]->imageProcessingSettings;
        R3DDecoder::ReleaseDecodeJob( jobs[i] );
    }
    jobs.clear();
    alignedSizes.clear();


	//Release the R3DDecoder instance
	R3DDecoder::ReleaseDecoder(decoder);
    FinalizeSdk();
	return 0;
}
