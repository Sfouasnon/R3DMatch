/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* Sample to display R3D clip metadata */

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

    // this sample uses a namespace again

    // first form to load a clip, this will try to load the clip and set
    // the class status to indicate succes or failure. In this scenario
    // you don't really have to worry about cleaning up the resources since
    // the destructor will do it as soon as 'clip' below goes out of scope
    Clip *clip = new Clip(argv[1]);

    // check to see if the R3D file was loaded succesfully, exit and
    // display an error message if it failed
    if (clip->Status() != LSClipLoaded)
    {
        printf("Error loading '%s'\n", argv[1]);
        delete clip;
        FinalizeSdk();
        return -2;
    }

    // display all available metadata
    for (size_t i = 0U; i < clip->MetadataCount(); i++)
    {
        printf("%02u: %s = %s\n", static_cast<unsigned int>(i + 1U),
            clip->MetadataItemKey(i).c_str(),		// get name of metadata item as std::string, convert to char *
            clip->MetadataItemAsString(i).c_str());	// get value of metadata item as std::string, convert to char *
    }
    
    printf("\n%u metadata items found\n", static_cast<unsigned int>(clip->MetadataCount()));
    delete clip;
    //Finalize the R3DSDK after destroying all R3DSDK objects
    FinalizeSdk();
	return 0;
}

