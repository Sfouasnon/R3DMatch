/* R3D SDK library version 9.2 header file. Do *NOT* use this
   header file with any other version of the R3D SDK library!
   
   This header file and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All
   rights reserved. Redistribution of this header is prohibited!
*/

#ifndef R3DSDK_CUSTOMIO_H
#define R3DSDK_CUSTOMIO_H

/* 
APIs that ignore a custom I/O interface:
	- InitializeSdk							(pathToDynamicLibraries must be accessible using normal OS APIs, and init needs to be called before SetIoInterface)
	- REDCL::REDCL							(compiledKernelCacheFolder should be located on local drive for performance reasons)
	- R3DDecoderOptions::setScratchFolder	(path should be located on a local drive for performance reasons)

APIs currently unsupported:
	- SaveRWGLog3G10ToOutputTransform3DLut
	- CreateRWGLog3G10ToOutputTransform3DLut
	- CreateRWGLog3G10ToOutputTransformLuts

APIs with full support:
	- Clip::Clip
	- Clip::LoadFrom
	- Clip::GetDefaultImageProcessingSettings
	- Clip::GetRmdHdrProcessingSettings
	- Clip::GetRmdXmp
	- Clip::CreateOrUpdateRmd
	- Clip::CreateTrimFrom
	- Clip::CreateSnapshotFrom
	- Clip::Load3DLut
	- Load3DLut
	- R3DStream::R3DStream

APIs that just return information and do not use I/O:
	- Clip::RmdSidecarPresent
	- Clip::GetRmdPath
	- Get3DLutPath
	
NOTE: paths passed in to the above APIs are mostly passed through as-is to your overridden IOInterface
      class. The one exception is that the end of the filename can and will get updated to open additional
	  R3D files and sidecar files such as RMD files (ie, *_001.R3D -> *_002.R3D, *.RMD, etc.). Follow
	  these guidelines to make sure this works correctly:
	  
	  1) paths must be valid C-style strings that are terminated with a NULL ('\0') character
	  
	  2) filenames must be separated from the path (if there is one) by at least one standard path separator
	     right in front of the filename (forward '/' or backward '\' slash)
		 
	  3) any other non-NULL characters can be part of the path / string in front of the filename. These do
	     not have to be UTF-8 compliant

	  Example of a valid non-standard C path: "#6f5902ac237024bdd0c176cb93063dc4#/A001_C001_AB0102_001.R3D"
	  
	  For the above example the following open calls could come in to your IOInterfaces instance:
		"#6f5902ac237024bdd0c176cb93063dc4#/A001_C001_AB0102_001.R3D"
		"#6f5902ac237024bdd0c176cb93063dc4#/A001_C001_AB0102_000.R3D"
		"#6f5902ac237024bdd0c176cb93063dc4#/A001_C001_AB0102_002.R3D"
		"#6f5902ac237024bdd0c176cb93063dc4#/A001_C001_AB0102.RMD"
		etc.		
*/

namespace R3DSDK {

class IOInterface
{
public:
	IOInterface();
	virtual ~IOInterface();

	// openening a file will be for EITHER read OR write access.
	enum FileAccess
	{
		IO_READ  = 1,	// open file ONLY for read access
		IO_WRITE = 2	// open file ONLY for write access
	};

	// user defined file handle. Do not use 0 (HANDLE_ERROR) or -1 (HANDLE_FALLBACK) for valid handles!
	typedef void * Handle;

	// ( 0) = file open failed, instructs SDK to not fallback to internal I/O
	// use this if the file does not exist or there was some error opening it
	static const Handle HANDLE_ERROR;

	// (-1) = overriding class can't open, instructs SDK to fallback to internal I/O (success not guaranteed)
	// use this if there's unsupported features (custom interface only supports IO_READ or IO_WRITE for example)
	static const Handle HANDLE_FALLBACK;

	// Open file for read OR write. Return HANDLE_FALLBACK if access mode is not supported.
	// 
	// Notes:
	// - file must be opened in binary mode
	// - file must be seeked to the beginning for both read or write modes
	// - write mode may delete the file first if it already exists
	// - requests will come in for non-existing files, return HANDLE_ERROR in these cases
	// - paths supplied to public SDK APIs will be passed through as-is, with the exception of the
	//   tails which may have been updated (ie, *_001.R3D -> *_002.R3D, etc.)
	// - if failure is returned then no other functions on this class will be called for this file
	//
	// Return values:
	// - return HANDLE_ERROR if file does not exist, or there was some error opening the file
	// - return HANDLE_FALLBACK if you wish the SDK to try and open the file instead (success not guaranteed)
	// - otherwise return a unique value (other than the above two) to identify the opened file in subsequent calls
	virtual Handle Open(const char * utf8Path, FileAccess access) = 0;

	// Return size of the file that belongs to the given handle.
	virtual unsigned long long Filesize(Handle handle) = 0;

	// Close the file that belongs to the given handle
	//
	// Notes:
	// - SDK may close files and then re-open them later, this is normal behavior
	// - will never get called with a handle value of HANDLE_ERROR or HANDLE_FALLBACK
	virtual void Close(Handle handle) = 0;

	// Read bytes from the file that belongs to the given handle.
	//
	// Notes:
	// - read offsets & sizes can and will be unaligned
	// - read sizes can and will be mixed (small and large)
	// - reads may or may not be sequential
	// - will only be called for files opened succesfully with IO_READ access mode
	//
	// Return values:
	// - return true if full requested size was read
	// - return false in all other cases
	virtual bool Read(void * outBuffer, size_t bytes, unsigned long long offset, Handle handle) = 0;

	// Sequentially write bytes to the file that belongs to the given handle.
	//
	// Notes:
	// - write sizes can and will be unaligned
	// - write sizes can and will be mixed (small and large)
	// - writes are always sequential and append, but after initial file open the file offset MUST be 0
	// - will only be called for files opened succesfully with IO_WRITE access mode
	//
	// Return values:
	// - return true if full requested size was written
	// - return false in all other cases
	virtual bool Write(const void * inBuffer, size_t bytes, Handle handle) = 0;

	// Create output folder if needed
	//
	// Notes:
	// - only called for camera streaming
	// - called before Open(IO_WRITE) gets calleed
	// - utf8Path is StreamCreate's path parameter + RDC folder name
	//
	// Return values:
	// - return true if path was created or no create is needed (path already exists)
	// - return false if normal platform mkdir() should be used
	virtual bool CreatePath(const char * utf8Path) = 0;
};

// Set or clear a custom I/O interface
//
// Notes:
// - make sure to close ALL clips before calling one of these functions, otherwise crashes,
//   errors and unpredictable behavior will happen!
// - custom I/O interface should be around from the call to SetIoInterface, as long as files
//   are open and ResetIoInterface or SetIoInterface with a new interface has not been called.
void SetIoInterface(IOInterface * io);
void ResetIoInterface();

}

#endif
