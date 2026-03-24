/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* Sample to demonstrate displaying R3D SDK version information */

#include <stdio.h>

#include "R3DSDK.h"

using namespace R3DSDK;

int main(int argc, char * argv[])
{
	// Initialize the R3DSDK prior to using any R3DSDK objects.
	InitializeStatus status = InitializeSdk(".", OPTION_RED_NONE);

	if ( status != ISInitializeOK)
	{
		printf("Failed to initialize SDK: %d\n", status);
		return 1;
	}

	printf("SDK VERSION: %s\n",  GetSdkVersion());
		
	// Finalize the R3DSDK after destroying all R3DSDK objects
	FinalizeSdk();
	return 0;
}

