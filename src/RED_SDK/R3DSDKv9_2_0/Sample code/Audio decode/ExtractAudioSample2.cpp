/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* Sample to store all audio in a .au file. Unlike the previous
   sample this one does not use the audio block interface but
   the sample interface. This allows random seeking in the
   audio track and retrieving a certain number of samples. The
   sample uses a 1MB buffer to store the samples in. 
   
   The output of this sample is exactly the same as the previous
   sample code that iterates through the audio blocks. */

#include <stdio.h>
#include <stdlib.h>
#include "R3DSDK.h"

using namespace R3DSDK;

// The R3D SDK requires that the output buffer is 512-byte aligned
unsigned char * AlignedMalloc(size_t & sizeNeeded)
{
	// alloc 511 bytes more to make sure we can align the buffer in case it isn't
	unsigned char * buffer = (unsigned char *)malloc(sizeNeeded + 511U);

	if (!buffer)
		return NULL;

	sizeNeeded = 0U;

	// cast to a 32-bit or 64-bit (depending on platform) integer so we can do the math
	uintptr_t ptr = (uintptr_t)buffer;

	// check if it's already aligned, if it is we're done
	if ((ptr % 512U) == 0U)
		return buffer;

	// calculate how many bytes we need
	sizeNeeded = 512U - (ptr % 512U);

	return buffer + sizeNeeded;
}

int main(int argc, char * argv[])
{
	// path to an R3D file & raw output file needs to be supplied on the command line
	if (argc != 3)
	{
		printf("Usage: %s sample.R3D out.au\n", argv[0]);
		printf("\nout.au will be overwritten without warning if it exists already!\n");
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

	// check if there's actually audio in the file
	size_t maxAudioBlockSize = 0U;
	size_t blocks = clip->AudioBlockCountAndSize(&maxAudioBlockSize);

	if (blocks == 0U)
	{
		clip->Close();
		printf("but no audio is present, aborting!\n");
        delete clip;
        FinalizeSdk();
		return -2;
	}

	unsigned int samplerate = clip->MetadataItemAsInt(RMD_SAMPLERATE);
	unsigned int samplesize = clip->MetadataItemAsInt(RMD_SAMPLE_SIZE);
	size_t channels = clip->AudioChannelCount();	// new in the 1.6 SDK release

	printf("contains %u %u-bit channels at %u Hz\n",static_cast<unsigned int>(channels), samplesize, samplerate);

	// retrieve the number of samples per channel
	unsigned long long samples = clip->AudioSampleCount();
	printf("Total number of samples per channel: %llu\n", samples);

	// make a copy for AlignedMalloc (it will change it)
	const size_t bufferSize = 1 * 1024 * 1024;	// 1MB audio buffer
	size_t adjusted = bufferSize;

	// alloc this memory 512-byte aligned
	unsigned char * audiobuffer = AlignedMalloc(adjusted);
	
	if (audiobuffer == NULL)
	{
		clip->Close();
		printf("Failed to allocate %u bytes of memory for output image\n", static_cast<unsigned int>(bufferSize));
        delete clip;
        FinalizeSdk();
		return -3;
	}

	printf("Writing audio to %s\n", argv[2]);

	// try to create output file using good ol' C I/O
	FILE * fout = fopen(argv[2], "wb");

	if (fout == NULL)
	{
		// free the original pointer, not the one adjusted for alignment
		free(audiobuffer - adjusted);

		printf("Error creating output file %s\n", argv[2]);
        delete clip;
        FinalizeSdk();
		return -2;
	}

	// construct the au header
	memset(audiobuffer, 0, 28);		// a lot of the buffer can be zero
	audiobuffer[0] = '.'; audiobuffer[1] = 's'; audiobuffer[2] = 'n'; audiobuffer[3] = 'd';
	audiobuffer[7] = 28;		// header size
	audiobuffer[8] = audiobuffer[9] = audiobuffer[10] = audiobuffer[11] = 0xFF;
	audiobuffer[15] = 5;	// 32-bit linear PCM
	audiobuffer[18] = samplerate >> 8; audiobuffer[19] = samplerate & 0xFF;
	audiobuffer[23] = channels & 0xFF;

	fwrite(audiobuffer, 1, 28, fout);

	// now loop through the audio track requesting 1MB of samples each time
	size_t bytesWritten = 28U;
	size_t samplesInBuffer;
	unsigned long long startSample = 0ULL;	// start at the first sample

	// simply loop until all the samples have been read, the last call to
	// DecodeAudio might return less bytes than requested (what the buffer
	// can hold).
	for (; startSample < samples; startSample += samplesInBuffer)
	{
		printf("Writing audio starting from sample %llu\n", startSample);
		
		// this sample application has a fixed size buffer, to calculate the
		// number of samples *per channel* that can fit inside we divide the
		// buffer size by the number of channels times the bytes per sample (4)
		samplesInBuffer = bufferSize / (channels * 4U);
		
		// call updates samplesInBuffer to reflect how many samples were actually
		// written into the buffer. Do NOT assume this is the same as what was
		// requested. It might be less at the end of the clip->
		clip->DecodeAudio(startSample, &samplesInBuffer, audiobuffer, bufferSize);

		// samplesInBuffer has been updated with the actual amount of samples
		// per channel that has been written into the supplied output buffer
		
		if (samplesInBuffer > 0U)
		{
			fwrite(audiobuffer, 1, samplesInBuffer * channels * 4U, fout);
			bytesWritten += samplesInBuffer * channels * 4U;
		}
	}

	fclose(fout);

	printf("Written %u bytes to %s.\nUse QuickTime player (for example) to play the file!\n", static_cast<unsigned int>(bytesWritten), argv[2]);

	// free the original pointer, not the one adjusted for alignment
	free(audiobuffer - adjusted);
    delete clip;
    //Finalize the R3DSDK after destroying all R3DSDK objects
    FinalizeSdk();
	return 0;
}

