/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* Sample to CRC check frames in a clip */

#include <stdio.h>

#include "R3DSDK.h"


using namespace R3DSDK;

static int CheckFrames(Clip * clip);

int maincheck(int argc, char * argv[])
{
	// path to an R3D file needs to be supplied on the command line
	if (argc != 2)
	{
		printf("Usage: %s sample.R3D\n", argv[0]);
		return -1;
	}
    
    //Initialize the R3DSDK prior to using any R3DSDK objects.
    InitializeStatus status = InitializeSdk(".", OPTION_RED_NONE);
    if (status != ISInitializeOK)
    {
        printf("Failed to initialize SDK: %d\n", status);
        return 1;
    }

	Clip * clip = new Clip(argv[1]);

	if (clip->Status() != R3DSDK::LSClipLoaded)
	{
		printf("Error loading '%s'\n", argv[1]);
        delete clip;
        FinalizeSdk();
		return -2;
	}

	const int failedFrames = CheckFrames(clip);

	if (failedFrames < 0)
		printf("Error: clip does not have CRCs\n");
	else if (failedFrames == 0)
		printf("%u frame(s) OK\n", (unsigned int)clip->VideoFrameCount());
	else
		printf("Error: %d frame(s) failed CRC check\n", failedFrames);

    delete clip;
    //Finalize the R3DSDK after destroying all R3DSDK objects
    FinalizeSdk();
	return 0;
}

static int CheckFrames(Clip * clip)
{
	int failed = 0;

	for (size_t f = 0; f < clip->VideoFrameCount(); f++)
	{
		const auto result = clip->CheckFrame(f);

		if (result == DSUnsupportedClipFormat)
		{
			return -1;
		}
		else if (result == DSDecodeFailed)
		{
			printf("Error: CRC failure for frame %u\n", (unsigned int)f);
			failed++;
		}
		else if (result != DSDecodeOK)
		{
			printf("Error: unexpected result %d\n", result);
			return -1;
		}
	}

	return failed;
}
