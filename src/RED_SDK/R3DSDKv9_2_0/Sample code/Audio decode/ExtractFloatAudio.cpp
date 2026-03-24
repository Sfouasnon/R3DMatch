/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* Sample to store all audio in a .au file */

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
	if (status != ISInitializeOK)
	{
		printf("Failed to initialize SDK: %d (%s)\n", status, GetSdkVersion());
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
	unsigned int channelmask = clip->MetadataItemAsInt(RMD_CHANNEL_MASK);

	// transform the channel mask into channel count
	// (SDK release 1.6 added clip->AudioChannelCount() function to do this)
	size_t channels = 0U;

	for (size_t t = 0U; t < 4U; t++)
	{
		if (channelmask & (1U << t))
			channels++;
	}

	const bool isFloat = (clip->MetadataItemAsInt(RMD_AUDIO_FORMAT) == 1);
	const char* format = isFloat ? "float" : "integer";

	printf("contains %u %u-bit channels of %s audio at %u Hz\n", static_cast<unsigned int>(channels), static_cast<unsigned int>(samplesize), format, static_cast<unsigned int>(samplerate));

	if (!isFloat)
	{
		printf("Error: clip does not have floating-point audio\n");
		return -1;
	}

	// make a copy for AlignedMalloc (it will change it)
	size_t adjusted = maxAudioBlockSize;

	// alloc this memory 512-byte aligned
	unsigned char * audiobuffer = AlignedMalloc(adjusted);
	
	if (audiobuffer == NULL)
	{
		clip->Close();
		printf("Failed to allocate %u bytes of memory for output image\n", static_cast<unsigned int>(maxAudioBlockSize));
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
	memset(audiobuffer, 0, 28);		// a lot of the header can be zero
	audiobuffer[0] = '.'; audiobuffer[1] = 's'; audiobuffer[2] = 'n'; audiobuffer[3] = 'd';
	audiobuffer[7] = 28;			// header size
	audiobuffer[8] = audiobuffer[9] = audiobuffer[10] = audiobuffer[11] = 0xFF;
	audiobuffer[15] = 6;			// 6 = 32-bit IEEE floating point
	audiobuffer[18] = samplerate >> 8; audiobuffer[19] = samplerate & 0xFF;
	audiobuffer[23] = channels & 0xFF;

	fwrite(audiobuffer, 1, 28, fout);

	// now loop through all the audio blocks and add them
	size_t bytesWritten = 28U;
	size_t bufferSize;

	for (size_t al = 0U; al < blocks; al++)
	{
		printf("Writing audio block %u/%u\n", static_cast<unsigned int>(al + 1U),static_cast<unsigned int>( blocks));
		
		bufferSize = maxAudioBlockSize;
		
		clip->DecodeFloatAudioBlock(al, audiobuffer, &bufferSize);	// bufferSize gets updated
		
		fwrite(audiobuffer, 1, bufferSize, fout);
		bytesWritten += bufferSize;
	}

	fclose(fout);

	printf("Written %u bytes to %s.\nUse foobar2000 (for example) to play the floating-point file!\n", static_cast<unsigned int>(bytesWritten), argv[2]);

	// free the original pointer, not the one adjusted for alignment
	free(audiobuffer - adjusted);
	delete clip;
	//Finalize the R3DSDK after destroying all R3DSDK objects
	FinalizeSdk();
	return 0;
}

