/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* Sample to display R3D clip properties */

#include <stdio.h>

#include "R3DSDK.h"


using namespace R3DSDK;

int main(int argc, char * argv[])
{
	// path to an R3D file needs to be supplied on the command line
	if (argc != 2)
	{
		printf("Usage: %s sample.R3D\n", argv[0]);
		return -1;
	}
    
    //Initialize the R3DSDK prior to using any R3DSDK objects.
    InitializeStatus status = InitializeSdk(".", OPTION_RED_NONE);
    if ( status != ISInitializeOK)
    {
        printf("Failed to initialize SDK: %d\n", status);
        return 1;
    }

	// unlike the previous example we don't have a "using namespace R3DSDK;"
	// at the top of this file. In this case we need to add the namespace
	// to each thing we need to use from the header files

	// first form to load a clip, this will try to load the clip and set
	// the class status to indicate succes or failure. In this scenario
	// you don't really have to worry about cleaning up the resources since
	// the destructor will do it as soon as 'clip' below goes out of scope
	R3DSDK::Clip *clip = new Clip(argv[1]);

	// check to see if the R3D file was loaded succesfully, exit and
	// display an error message if it failed
	if (clip->Status() != R3DSDK::LSClipLoaded)
	{
		printf("Error loading '%s'\n", argv[1]);
        delete clip;
        FinalizeSdk();
		return -2;
	}

	size_t maxAudioBlockSize;

	// display the clip properties
	printf("Input file                  : %s\n", argv[1]);
	printf("Number of video tracks      : %u\n", static_cast<unsigned int>(clip->VideoTrackCount()));
	printf("Resolution                  : %u x %u\n", static_cast<unsigned int>(clip->Width()), static_cast<unsigned int>(clip->Height()));
	printf("Video framerate             : %.3f fps\n", clip->VideoAudioFramerate());
	printf("Timecode framerate          : %.3f fps\n", clip->TimecodeFramerate());
	printf("Number of video frames      : %u\n", static_cast<unsigned int>(clip->VideoFrameCount()));
	printf("Number of raw audio blocks  : %u\n", static_cast<unsigned int>(clip->AudioBlockCountAndSize(&maxAudioBlockSize)));
	printf("Maximum raw audio block size: %u\n", static_cast<unsigned int>(maxAudioBlockSize));
	printf("Start absolute timecode     : %s\n", clip->AbsoluteTimecode(0U));
	printf("Ending absolute timecode    : %s\n", clip->AbsoluteTimecode(clip->VideoFrameCount() - 1U));
	printf("Start edge timecode         : %s\n", clip->EdgeTimecode(0U));
	printf("Ending edge timecode        : %s\n", clip->EdgeTimecode(clip->VideoFrameCount() - 1U));

    delete clip;
    //Finalize the R3DSDK after destroying all R3DSDK objects
    FinalizeSdk();
	return 0;
}

