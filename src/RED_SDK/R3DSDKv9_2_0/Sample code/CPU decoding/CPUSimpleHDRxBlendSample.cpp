/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* HDRx simple blend sample. Decode the first frame on the first
   (main) track and decode the first frame on the second (X)
   track and then do a simple blend between the two.
*/

#include <stdio.h>
#include <stdlib.h>
#include "R3DSDK.h"

using namespace R3DSDK;

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

// Do simple blend on 16-bit interleaved RGB data
void HDRSimpleBlend(unsigned short * buffer1, unsigned short * buffer2, size_t bufferSize, float bias);

int main(int argc, char * argv[])
{
    //Initialize the R3DSDK prior to using any R3DSDK objects.
    InitializeStatus status = InitializeSdk(".", OPTION_RED_NONE);
    if ( status != ISInitializeOK)
    {
        printf("Failed to initialize SDK: %d\n", status);
        return 1;
    }
    
	// path to an HDRx R3D file & raw output file needs to be supplied on the command line
	if ((argc != 3) && (argc != 4))
	{
		printf("Usage: %s sample.R3D out.raw [blend bias]\n", argv[0]);
		printf("\nout.raw will be overwritten without warning if it exists already!\n");
		printf("blend bias: optional, in range -10 to +10. Input value will be divided by 10.\n");
		printf("\t-10 only uses X image, +10 only uses main image. Default value of 0 uses a\n");
		printf("\t50%% blend between the main and X image\n");
        FinalizeSdk();
		return -1;
	}

	float bias = 0.0f;

	if (argc == 4)
	{
		int ibias = atoi(argv[3]);

		if ((ibias < -10) || (ibias > 10))
		{
			printf("Error: blend bias must be in the range -10 -- +10\n");
            FinalizeSdk();
			return -1;
		}

		bias = (float)ibias / 10.0f;
	}

	// load the clip
	Clip *clip = new Clip(argv[1]);

	// let the user know if this failed
	if (clip->Status() != LSClipLoaded)
	{
		printf("Error loading %s\n", argv[1]);
        delete clip;
        FinalizeSdk();
		return -1;
	}
	
	printf("Loaded %s\n", argv[1]);

	// make sure this clip is an HDRx clip
	if ((clip->VideoTrackCount() != 2U) || (clip->MetadataItemAsInt(RMD_HDR_MODE) != 2))
	{
		printf("Error: supplied clip is not an HDRx clip!\n");
        delete clip;
        FinalizeSdk();
		return -5;
	}

	// calculate how much ouput memory we're going to need
	size_t width = clip->Width() / 2U;
	size_t height = clip->Height() / 2U;	// going to do a half resolution decode

	// three channels (RGB) in 16-bit (2 bytes) requires this much memory:
	size_t memNeeded = width * height * 3U * 2U;

	// make a copy for AlignedMalloc (it will change it)
	size_t adjusted1 = memNeeded;
	size_t adjusted2 = memNeeded;

	// alloc this memory 16-byte aligned for both images
	unsigned char * imgbuffer1 = AlignedMalloc(adjusted1);
	unsigned char * imgbuffer2 = AlignedMalloc(adjusted2);
	
	if ((imgbuffer1 == NULL) || (imgbuffer2 == NULL))
	{
		printf("Failed to allocate %d bytes of memory for output image\n", static_cast<unsigned int>(memNeeded * 2U));
        if(imgbuffer1 == NULL)
        {
            free(imgbuffer1 - adjusted1);
        }
        if(imgbuffer2 == NULL)
        {
            free(imgbuffer2 - adjusted2);
        }
        delete clip;
        FinalizeSdk();
		return -3;
	}

	// create and fill out a decode job structure so the
	// decoder knows what you want it to do
	VideoDecodeJob job;

	// letting the decoder know how big the buffer is (we do that here
	// since AlignedMalloc below will overwrite the value in this
	job.OutputBufferSize = memNeeded;

	// we're going with the clip's default image processing
	// see the next sample on how to change some settings

	// decode at half resolution at very good but not premium quality
	job.Mode = DECODE_HALF_RES_GOOD;

	// store the image in a 16-bit interleaved RGB format
	job.PixelType = PixelType_16Bit_RGB_Interleaved;

	// decode the first frame (0) of the first (main) track (0)
	printf("Decoding frame 0 track 0 at %d x %d\n", static_cast<unsigned int>(width), static_cast<unsigned int>(height));

	// store the image here
	job.OutputBuffer = imgbuffer1;

	if (clip->VideoTrackDecodeFrame(0U /* track 0 */, 0U /* frame 0 */, job) != DSDecodeOK)
	{
		printf("Decode failed?\n");
        free(imgbuffer1 - adjusted1);
        free(imgbuffer2 - adjusted2);
        delete clip;
        FinalizeSdk();
		return -4;
	}

	// decode the first frame (0) of the second (X) track (1)
	printf("Decoding frame 0 track 1 at %d x %d\n", static_cast<unsigned int>(width), static_cast<unsigned int>(height));

	// store the image here
	job.OutputBuffer = imgbuffer2;

	if (clip->VideoTrackDecodeFrame(1U /* track 0 */, 0U /* frame 0 */, job) != DSDecodeOK)
	{
		printf("Decode failed?\n");
        free(imgbuffer1 - adjusted1);
        free(imgbuffer2 - adjusted2);
        delete clip;
        FinalizeSdk();
		return -4;
	}

	// do a simple blend and store the output in imgbuffer1
	printf("Blending the two images with a bias of %.1f\n", bias);
	
	HDRSimpleBlend((unsigned short *)imgbuffer1, (unsigned short *)imgbuffer2, memNeeded, bias);

	// write output to disk
	printf("Writing image to %s\n", argv[2]);

	// try to create output file using good ol' C I/O
	FILE * fout = fopen(argv[2], "wb");

	if (fout == NULL)
	{
		// free the original pointers, not the one adjusted for alignment
		free(imgbuffer1 - adjusted1);
		free(imgbuffer2 - adjusted2);

		printf("Error creating output file %s\n", argv[2]);
        delete clip;
        FinalizeSdk();
		return -2;
	}

	// write the image to disk
	fwrite(imgbuffer1, 1, memNeeded, fout);
	fclose(fout);

	printf("You can load the raw file in Photoshop, select %d x %d for the resolution\n", static_cast<unsigned int>(width), static_cast<unsigned int>(height));
	printf("with 3 components, interleaved, 16-bit with PC byte ordering\n");

	// free the original pointers, not the one adjusted for alignment
	free(imgbuffer1 - adjusted1);
	free(imgbuffer2 - adjusted2);
    delete clip;
    //Finalize the R3DSDK after destroying all R3DSDK objects
    FinalizeSdk();
	return 0;
}

// Do simple HDR blend on interleaved 16-bit RGB data
// if bias = -1.0 only buffer1 (main in this case) data will be used
// if bias =  0.0 50% of buffer1 and 50% of buffer2 data will be used
// if bias = +1.0 only buffer2 (X in this case) data will be used
// etc
void HDRSimpleBlend(unsigned short * buffer1, unsigned short * buffer2, size_t bufferSize, float bias)
{
	if ((buffer1 == NULL) || (buffer2 == NULL) || (bufferSize == 0U))
		return;

	// clamp bias to valid range
	if (bias < -1.0f)
	{
		bias = -1.0f;
	}
	else if (bias > 1.0f)
	{
		bias = 1.0f;
	}

	const float bias1 = 0.5f + bias / 2.0f;
	const float bias2 = 0.5f - bias / 2.0f;

	// no need to do red, green & blue seperate for this blend
	for (int i = 0; i < (int)bufferSize / 2; i++)
	{
		// get either R, G or B from both images
		const float pixel1 = (float)buffer1[0] / 65535.0f;
		const float pixel2 = (float)buffer2[0] / 65535.0f;

		// store blend result in first buffer
		buffer1[0] = (unsigned short)((pixel1 * bias1 + pixel2 * bias2) * 65535.0f);

		buffer1++;
		buffer2++;
	}
}
