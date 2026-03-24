/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* Sample to decode the first frame at half resolution fast */

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

int main(int argc, char * argv[])
{
	// path to an R3D file & raw output file needs to be supplied on the command line
	if (argc != 3)
	{
		printf("Usage: %s sample.R3D out.raw\n", argv[0]);
		printf("\nout.raw will be overwritten without warning if it exists already!\n");
		return -1;
	}

    //Initialize the R3DSDK prior to using any R3DSDK objects.
    InitializeStatus status = InitializeSdk(".", OPTION_RED_NONE);
    if ( status != ISInitializeOK)
    {
        printf("Failed to initialize SDK: %d\n", status);
        return 1;
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

	// calculate how much ouput memory we're going to need
	size_t width = clip->Width() / 2U;
	size_t height = clip->Height() / 2U;	// going to do a half resolution decode

	// three channels (RGB) in 16-bit (2 bytes) requires this much memory:
	size_t memNeeded = width * height * 3U * 2U;

	// make a copy for AlignedMalloc (it will change it)
	size_t adjusted = memNeeded;

	// alloc this memory 16-byte aligned
	unsigned char * imgbuffer = AlignedMalloc(adjusted);
	
	if (imgbuffer == NULL)
	{
		printf("Failed to allocate %d bytes of memory for output image\n", static_cast<unsigned int>(memNeeded));
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

	// store the image here
	job.OutputBuffer = imgbuffer;

	// store the image in a 16-bit planar RGB format
	job.PixelType = PixelType_16Bit_RGB_Planar;

	// decode the first frame (0) of the clip
	printf("Decoding image at %d x %d\n", static_cast<unsigned int>(width), static_cast<unsigned int>(height));

	if (clip->DecodeVideoFrame(0U, job) != DSDecodeOK)
	{
		printf("Decode failed?\n");
        delete clip;
        FinalizeSdk();
		return -4;
	}

	printf("Writing image to %s\n", argv[2]);

	// try to create output file using good ol' C I/O
	FILE * fout = fopen(argv[2], "wb");

	if (fout == NULL)
	{
		// free the original pointer, not the one adjusted for alignment
		free(imgbuffer - adjusted);

		printf("Error creating output file %s\n", argv[2]);
        delete clip;
        FinalizeSdk();
		return -2;
	}

	// write the image to disk
	fwrite(imgbuffer, 1, memNeeded, fout);
	fclose(fout);

	printf("You can load the raw file in Photoshop, select %d x %d for the resolution\n", static_cast<unsigned int>(width), static_cast<unsigned int>(height));
	printf("with 3 components, non-interleaved, 16-bit with PC byte ordering\n");

	// free the original pointer, not the one adjusted for alignment
	free(imgbuffer - adjusted);

    delete clip;
    //Finalize the R3DSDK after destroying all R3DSDK objects
    FinalizeSdk();
	return 0;
}

