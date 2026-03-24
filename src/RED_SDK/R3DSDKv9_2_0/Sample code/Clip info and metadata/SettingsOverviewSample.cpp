/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* Sample to display image processing options

   The importance of this sample is to show how you can use the
   SDK to get the information needed to present the different
   processing options in your user interface. This way you do
   not manually need to keep your application in sync whenever
   we add certain new processing features!
*/

#include <stdio.h>

#include "R3DSDK.h"

using namespace R3DSDK;

int main(int argc, char * argv[])
{
	// path to an R3D file
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
	printf("'clip -->' below indicates the clip's default setting\n");
	printf("'<-- default' below indicates the UI's default setting\n");

	// get the clip's default settings
	ImageProcessingSettings ip;
	clip->GetDefaultImageProcessingSettings(ip);
	size_t i;

	// display the list of ISO options
	printf("\nISO options:\n");

	for (i = 0U; i < ImageProcessingLimits::ISOCount; i++)
	{
		size_t curISO = ImageProcessingLimits::ISOList[i];

		if (curISO == ip.ISO)
			printf("clip -->");
		else
			printf("        ");

		printf("%u", static_cast<int>(curISO));

		if (curISO == ImageProcessingLimits::ISODefault)
			printf("<-- default");

		printf("\n");
	}

	// display the list of gamma curves
	printf("\nGamma curves:\n");

	for (i = 0U; i < ImageProcessingLimits::GammaCurveCount; i++)
	{
		ImageGammaCurve curGamma = ImageProcessingLimits::GammaCurveMap[i];

		if (curGamma == ip.GammaCurve)
			printf("clip -->");
		else
			printf("        ");

		printf("%s", ImageProcessingLimits::GammaCurveLabels[i]);

		if (curGamma == ImageProcessingLimits::GammaCurveDefault)
			printf("<-- default");

		printf("\n");
	}

	// display the list of color spaces
	printf("\nColor spaces:\n");

	for (i = 0U; i < ImageProcessingLimits::ColorSpaceCount; i++)
	{
		ImageColorSpace curSpace = ImageProcessingLimits::ColorSpaceMap[i];

		if (curSpace == ip.ColorSpace)
			printf("clip -->");
		else
			printf("        ");

		printf("%s", ImageProcessingLimits::ColorSpaceLabels[i]);

		if (curSpace == ImageProcessingLimits::ColorSpaceDefault)
			printf("<-- default");

		printf("\n");
	}

	printf("\n");

	// display the image processing parameters
	printf("brightness\t%f\t--\t%f\t--\t%f\n",
		ImageProcessingLimits::BrightnessMin,
		ImageProcessingLimits::BrightnessDefault,
		ImageProcessingLimits::BrightnessMax);

	printf("contrast\t%f\t--\t%f\t--\t%f\n",
		ImageProcessingLimits::ContrastMin,
		ImageProcessingLimits::ContrastDefault,
		ImageProcessingLimits::ContrastMax);

	printf("DRX\t\t%f\t--\t%f\t--\t%f\n",
		ImageProcessingLimits::DRXMin,
		ImageProcessingLimits::DRXDefault,
		ImageProcessingLimits::DRXMax);

	printf("exposure comp\t%f\t--\t%f\t--\t%f\n",
		ImageProcessingLimits::ExposureMin,
		ImageProcessingLimits::ExposureDefault,
		ImageProcessingLimits::ExposureMax);

	printf("RGB gains\t%f\t--\t%f\t--\t%f\n",
		ImageProcessingLimits::GainsMin,
		ImageProcessingLimits::GainsDefault,
		ImageProcessingLimits::GainsMax);

	printf("kelvin\t\t%f\t--\t%f\t--\t%f\n",
		ImageProcessingLimits::KelvinMin,
		ImageProcessingLimits::KelvinDefault,
		ImageProcessingLimits::KelvinMax);

	printf("tint\t\t%f\t--\t%f\t--\t%f\n",
		ImageProcessingLimits::TintMin,
		ImageProcessingLimits::TintDefault,
		ImageProcessingLimits::TintMax);

	printf("saturation\t%f\t--\t%f\t--\t%f\n",
		ImageProcessingLimits::SaturationMin,
		ImageProcessingLimits::SaturationDefault,
		ImageProcessingLimits::SaturationMax);

	printf("shadow\t\t%f\t--\t%f\t--\t%f\n",
		ImageProcessingLimits::ShadowMin,
		ImageProcessingLimits::ShadowDefault,
		ImageProcessingLimits::ShadowMax);

	printf("FLUT\t\t%f\t--\t%f\t--\t%f\n",
		ImageProcessingLimits::FLUTMin,
		ImageProcessingLimits::FLUTDefault,
		ImageProcessingLimits::FLUTMax);

    delete clip;
    //Finalize the R3DSDK after destroying all R3DSDK objects
    FinalizeSdk();
	return 0;
}

