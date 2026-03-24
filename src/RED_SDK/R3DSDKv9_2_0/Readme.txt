Welcome to v9.2.0 of the Mac, Windows & Linux R3D SDK!

For any feedback, bugs or questions please contact RED-R3DSDK@nikon.com


Important 9.2 notes
-----------------------------------------------------------
NOTE: 12 GB minimum VRAM recommended for GPU processing
NOTE: Metal requires v2.3 and uses ARC (Automatic Reference Counting)
NOTE: R3D SDK is 64-bit only


What's new for the 9.2.0 release
-----------------------------------------------------------
The 9.2.0 release contains these changes:
- Added: NIKKOR Z Lens Distortion Correction support for IPP2. See the new
         metadata RMD_LENS_DISTORTION_CORRECTION, ImageProcessingSettings::
         LensDistortionCorrection field and LensCorrection enum.
- Added: initial Windows ARM64 support with OpenCL image processing.
   NOTE: redistributable DLLs have the 'a64' suffix instead of 'x64'.
   NOTE: GPU decompression is not supported at this time. See GpuDecoder::DecodeSupportedForClip() API.
   NOTE: CUDA is not supported at this time (new error ISCudaNotAvailable will be returned).
- Improved: Broadcast gain & detail for clips shot with updated Broadcast firmware.
- Improved: black offset in GPU image processing with high gain settings in IPP2.

The 9.2.0 release contains bug fixes for:
- Fixed: rare crash when loading N-RAW or R3D NE clips.
- Fixed: rare crash when 3D LUT is applied with steep user curve.
- Fixed: non-SDR Broadcast decodes were too bright.


Additional links
-----------------------------------------------------------
Sample R3D clips   : https://www.red.com/sample-r3d-files
Sample R3D NE clips: https://www.nikonusa.com/content/Zcinema-raw-file-download
Sample N-RAW clips : https://www.nikonusa.com/content/z6iii-raw-video-downloads
                     https://www.nikonusa.com/content/z-8-raw-video-downloads
Broadcast clip:      https://downloads.red.com/software/sdk/R3D/Broadcast_sample_clip.zip

RED LUT KITS:
- https://www.red.com/download/3d-cude-luts-and-ipp2
- https://www.red.com/download/red-lut-kit

REDCINE-X PRO, REDline & more:
- https://www.red.com/download/redcine-x-pro-win & https://www.red.com/download/redcine-x-pro-win-beta
- https://www.red.com/download/redcine-x-pro-mac & https://www.red.com/download/redcine-x-pro-mac-beta
- https://www.red.com/download/redline-linux-beta 
- https://www.red.com/downloads


Introduction
-----------------------------------------------------------
The R3D SDK contains the following functionality:

- Compatible with RED DSMC3 firmware up to build 2.0.X
- Compatible with all RED DSMC/DSMC2 firmware
- Compatible with all RED ONE firmware
- Compatible with Nikon Z6 III, Z 8, Z 9 & ZR N-RAW (.NEV) & R3D NE (.R3D) clips
- Extracting clip properties and associated metadata
- Decoding video frames at various resolutions, qualities & with different image processing settings
- Decoding of audio samples and audio blocks
- Assistance for getting the right options & ranges for the image processing into your User Interface
- Auto White Balance from a user selectable point
- RMD sidecar support to update a clips look and have it stay with the clip
- 3D LUT (.CUBE) sidecar support
- Single or multi-GPU processing through CUDA, Metal or OpenCL


Hardware acceleration requirements
-----------------------------------------------------------
- CUDA: CUDA 6.5 runtime or higher, GPU compute capability 3.0 or higher
- OpenCL: 1.1 or higher. CPU OpenCL devices not supported
- Metal: v2.3 or higher, macOS 10.11 minimum. 10.14 or higher recommended for concurrent dispatch support
- Minimum of 12 GB of graphics card memory, more recommended
- NOTE: GPU acceleration is not available for HDRx blending or ColorVersion1
- NOTE: GPU decode (decompression) support not available for RED ONE, N-RAW or R3D NE clips, see GpuDecoder::DecodeSupportedForClip() API
- NOTE: GPU decode (decompression) & Metal image processing not supported by R3DDecoder
- NOTE: CUDA not supported on Windows ARM64


What is included
-----------------------------------------------------------
- Broadcast Image Pipeline.txt        : more detail information about the broadcast image processing pipe available for clips shot in broadcast mode
- How to use the SDK.txt              : how to use the SDK, installation & building your application
- IPP2 How to use.txt                 : how to use the newer IPP2 image processing pipe
- IPP2 Image Pipeline Stages.pdf      : high level overview of the IPP2 image processing pipe, its goals and the parts that make up IPP2
- Nikon N-RAW.txt                     : information on N-RAW features and SDK API impacts
- Nikon R3D NE.txt                    : information on R3D NE features and SDK API impacts
- OpenCL kernel caching.txt           : tips on avoiding lengthy OpenCL startup delays due to kernel compilation
- R3D SDK readme.txt                  : this document
- Release history.txt                 : release notes for all versions released
- SDK License Agreement.pdf           : license agreement for the R3D SDK
- White Paper on RWG and Log3G10.pdf  : White paper on and definitions for REDWideGamutRGB and Log3G10
- Include\R3DSDK.h                    : include this first to get access to R3D loading and procesing
- Include\R3DSDKCuda.h                : include to use RED CUDA processing in your existing CUDA pipeline
- Include\R3DSDKCustomIO.h            : include to replace the SDK I/O back-end with a custom implementation
- Include\R3DSDKDecoder.h             : include to use RED managed GPU processing using either CUDA or OpenCL
- Include\R3DSDKDefinitions.h         : various definitions  (no need to directly include this as R3DSDK.h does so)
- Include\R3DSDKMetadata.h            : metadata definitions (no need to directly include this as R3DSDK.h does so)
- Include\R3DSDKMetal.h               : include to use RED Metal  processing in your existing Apple Metal pipeline
- Include\R3DSDKOpenCL.h              : include to use RED OpenCL processing in your existing OpenCL pipeline
- Include\R3DSDKStream.h              : include to process incoming camera network stream and save to R3D clips
- Lib\linux64\libR3DSDK*.a            : Linux 64-bit Intel static libraries that must be linked to your project.
- Lib\mac64\libR3DSDK-libcpp.a        : Mac Universal 64-bit Intel/Apple Silicon static library that must be linked to your project.
- Lib\win64\R3DSDK-*.lib              : Windows 64-bit Intel static libraries that must be linked to your project.
- Lib\winarm64\R3DSDK-*.lib           : Windows 64-bit ARM static libraries that must be linked to your project.
- Redistributable\linux\              : Dynamic libraries (.so) you must distribute with your Linux application
- Redistributable\mac\                : Dynamic libraries (.dylib) you must distribute with your macOS application
- Redistributable\win\                : Dynamic libraries (.DLL) you must distribute with your Windows application
- Sample code\Audio decode\           : decode audio in blocks or samples in integer or float
- Sample code\Clip info and metadata\ : get clip properties & metadata and check files for CRC errors
- Sample code\CPU decoding\           : basic CPU only decoding, image processing and HDRx blending
- Sample code\Creating clips\         : create new shorter R3D clips and stream compressed R3D data from a camera over the network
- Sample code\Custom IO\              : replace the SDK I/O back-end with a custom implementation
- Sample code\GPU decoding\           : CUDA, OpenCL and Metal GPU accelerated decoding
- Sample code\GetSdkVersionSample.cpp : simple example to initialize SDK & display version information (good info to log!)


This software was developed using KAKADU software.


(c) 2008-2026 RED DIGITAL CINEMA. All rights reserved.
