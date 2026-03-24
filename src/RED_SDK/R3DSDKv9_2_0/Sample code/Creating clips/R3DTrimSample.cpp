/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* Normal and snapshot trim. Trim creates a new R3D from an
   existing R3D clip with only the selected range of frames.
   Trimming currently removes any audio present. Trim does
   support HDRx (frames from both tracks will be included).
   Snapshot trim creates a single frame R3D with a given name.
*/

#include <stdio.h>

#include "R3DSDK.h"

using namespace R3DSDK;

// flag to synchronize the callback with the main thread, for this sample we simply
// use a 4-byte aligned integer instead of a more propery synchronization system
volatile int trimDone = 0;

bool TrimCallback(CreateStatus status, void * privateData, size_t frameNoDone, size_t lastFrameToDo)
{
	// CSStarted, CSRequestOutOfRange, CSInvalidParameter, CSInvalidSourceClip & CSInvalidPath
	// should not be returned in the callback. These can only be returned from CreateTrimFrom()

	switch (status)
	{
		case CSFrameAdded: printf("Frame %u/%u added\n", (unsigned int)frameNoDone + 1U, (unsigned int)lastFrameToDo); break;
		case CSDone: printf("All done!\n"); break;

		case CSOutOfMemory: printf("Error: out of memory\n"); break;
		case CSFailedToGetSourceFrame: printf("Error: reading from source\n"); break;
		case CSFailedToCreateDestination: printf("Error: creating output\n"); break;
		case CSFailedToWriteToDestination: printf("Error: writing to output\n"); break;
		case CSUnknownError: printf("Error: unknown error, this should not happen\n"); break;

		default: printf("Error: unexpected status: %u\n", status); break;
	}

	if (status != CSFrameAdded)
	{
		// trim is done, or an error occurred
		trimDone = 1;
	}

	// return true to proceed with the trim, or false the trim should be aborted
	return true;
}

int main(int argc, char * argv[])
{
	// path to an HDRx R3D file & raw output file needs to be supplied on the command line
	if (argc != 3)
	{
		printf("Usage: %s sample.R3D existing_empty_directory\n", argv[0]);
		printf("\nsample.R3D must have more than 10 frames\n");
		printf("supplied output directory must be empty!\n");
		printf("a snapshot R3D will be created in the current directory\n");
		return -1;
	}

    //Initialize the R3DSDK prior to using any R3DSDK objects.
    InitializeStatus init_status = InitializeSdk(".", OPTION_RED_NONE);
    if ( init_status != ISInitializeOK)
    {
        printf("Failed to initialize SDK: %d\n", init_status);
        return 1;
    }
    
	Clip *clip = new Clip();

	if (clip->LoadFrom(argv[1]) != LSClipLoaded)
	{
		printf("Error: failed to load clip '%s'\n", argv[1]);
        delete clip;
        FinalizeSdk();
		return -2;
	}

	printf("Frames in source clip: %u\n", (unsigned int)clip->VideoFrameCount());

	if (clip->VideoFrameCount() <= 10U)
	{
		printf("Error: this sample requires a clip with more than 10 frames\n");
        clip->Close();
        delete clip;
        FinalizeSdk();
		return -3;
	}

	CreateStatus status;

	printf("Starting 10 frame trim operation\n");
    
    
    //trim including audio.
	if ((status = Clip::CreateTrimFrom(*clip, argv[2], 0U, 9U, true, NULL, TrimCallback)) != CSStarted)
	{
		printf("Error starting trim: ");

		switch (status)
		{
			case CSRequestOutOfRange: printf("request out of range\n"); break;
			case CSInvalidParameter: printf("invalid parameter\n"); break;
			case CSInvalidSourceClip: printf("trim does not support RED ONE clips shot on firmware build 15 or below\n"); break;
			case CSInvalidPath: printf("output path is invalid (see trim requirements)\n"); break;
			default: printf("Error: unexpected status %u\n", status); break;
		}
	}
	else
	{
		// let's wait for the trim to be done, normally you would want to
		// use some sort of event system instead of polling
		while (trimDone == 0)
		{
		};

		printf("10 frame trim complete\n");
	}

	printf("Creating a snapshot trim of frame 0\n");

	if ((status = Clip::CreateSnapshotFrom(*clip, "frame0_snapshot.R3D", 0)) != CSDone)
	{
		printf("Error creating snapshot %u\n", status);
	}
	else
	{
		printf("Snapshot trim complete\n");
	}

    clip->Close();
    delete clip;
    FinalizeSdk();
	return 0;
}
