/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* Sample to demonstrate replacing the SDK I/O back-end with a
   custom implementation. This can be used to access clips
   stored in the cloud or certain storage systems etc. */

#include <stdio.h>

#include "R3DSDK.h"
#include "R3DSDKCustomIO.h"

using namespace R3DSDK;

class IOSample : public IOInterface
{
public:
	IOSample(/* can have custom arguments here */)
	{
	}

	virtual Handle Open(const char * utf8Path, FileAccess access)
	{
		// this sample class does not handle write, let SDK handle that
		if (access == IO_WRITE)
			return HANDLE_FALLBACK;

		// note: ALL files MUST be opened in binary mode
		FILE * f = fopen(utf8Path, "rb");

		// open failed, no reason to let SDK try it as it would fail as well
		if (f == NULL)
			return HANDLE_ERROR;

		return reinterpret_cast<Handle>(f);
	}

	virtual void Close(Handle handle)
	{
		fclose(reinterpret_cast<FILE *>(handle));
	}

	virtual unsigned long long Filesize(Handle handle)
	{
		FILE * f = reinterpret_cast<FILE *>(handle);

#ifdef _WIN32
		const long long backup = _ftelli64(f);
		fseek(f, 0, SEEK_END);
		const unsigned long long fsize = _ftelli64(f);
		_fseeki64(f, backup, SEEK_SET);
#else
		const off_t backup = ftello(f);
		fseeko(f, 0LL, SEEK_END);
		const unsigned long long fsize = (unsigned long long)ftello(f);
		fseeko(f, backup, SEEK_SET);
#endif

		return fsize;
	}

	virtual bool Read(void * outBuffer, size_t bytes, unsigned long long offset, Handle handle)
	{
		FILE * f = reinterpret_cast<FILE *>(handle);

#ifdef _WIN32
		if (_fseeki64(f, offset, SEEK_SET) != 0)
			return false;
#else
		if (fseeko(f, offset, SEEK_SET) != 0)
			return false;
#endif

		return (fread(outBuffer, bytes, 1, f) == 1);
	}

	// will never get called due to HANDLE_FALLBACK in Open(),
	// but must still be implemented to satisfy the interface
	virtual bool Write(const void * inBuffer, size_t bytes, Handle handle)
	{
		return false;
	}

private:
	// store private data here if needed, this should likely NOT be
	// tied to a specific file (outside of caching) as this class
	// instance will be shared with multiple files.
};

int main(int argc, char * argv[])
{
	if (InitializeSdk(".", 0) != ISInitializeOK)
	{
		printf("Failed to initialize SDK\n");
		return -1;
	}
	
	IOSample * sample = new IOSample();
	SetIoInterface(sample);

	// scope the Clip class so it gets destroyed and file handle(s) closed before the call to ResetIoInterface()
	{
		Clip clip(argv[1]);
		
		if (clip.Status() != LSClipLoaded)
		{
			printf("Error loading clip %s\n", argv[1]);
			return -1;
		}
		
		printf("Clip %d x %d with %d frames\n", (int)clip.Width(), (int)clip.Height(), (int)clip.VideoFrameCount());
	}
	
	ResetIoInterface();
	delete sample;
	
	FinalizeSdk();
	
	return 0;
}

