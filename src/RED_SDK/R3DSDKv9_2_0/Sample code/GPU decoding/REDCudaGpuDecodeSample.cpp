/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

#include <R3DSDK.h>
#include <R3DSDKCuda.h>
#include <R3DSDKDefinitions.h>

#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mutex>
#include <thread>
#include <condition_variable>
#include <list>
#include <map>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <time.h>
#include <iostream>

#define PER_GPU_QUEUE_SIZE	4
#define TOTAL_FRAMES		1000
#define GPU_THREADS			1		// set to 2 for 2 GPUs, etc.
#define NUM_STREAMS			4
#define FRAME_QUEUE_SIZE	(PER_GPU_QUEUE_SIZE * GPU_THREADS)

namespace
{
	R3DSDK::GpuDecoder * GPU_DECODER;
	R3DSDK::REDCuda * RED_CUDA;

	int CUDA_DEVICE_ID = 0;
	volatile int cpuDone = 0;
	volatile int gpuDone = 0;
}

class SimpleMemoryPool
{
public:
	static SimpleMemoryPool * getInstance()
	{
		static SimpleMemoryPool * instance = NULL;

		if (instance == NULL)
		{
			std::unique_lock<std::mutex> lock(guard);
			if (instance == NULL)
			{
				instance = new SimpleMemoryPool();
			}
		}
		return instance;
	}

	static cudaError_t cudaMalloc(void ** p, size_t size)
	{
		return getInstance()->malloc_d(p, size);
	}

	static cudaError_t cudaFree(void * p)
	{
		return getInstance()->free_d(p);
	}

	static cudaError_t cudaMallocArray(
		struct cudaArray ** 	array,
		const struct cudaChannelFormatDesc * 	desc,
		size_t 	width,
		size_t 	height = 0,
		unsigned int 	flags = 0)
	{
		return getInstance()->malloc_array(array,
			desc,
			width,
			height,
			flags);
	}

	static cudaError_t cudaMalloc3DArray(
		struct cudaArray ** 	array,
		const struct cudaChannelFormatDesc * 	desc,
		struct cudaExtent ext,
		unsigned int 	flags = 0)
	{
		return getInstance()->malloc_array_3d(array,
			desc,
			ext,
			flags);
	}

	static cudaError_t cudaFreeArray(cudaArray * p)
	{
		getInstance()->free_array(p);

		return cudaSuccess;
	}

	static cudaError_t cudaMallocHost(void ** p, size_t size)
	{
		return getInstance()->malloc_h(p, size);
	}

	static cudaError_t cudaHostAlloc(void ** p, size_t size, unsigned int flags)
	{
		return getInstance()->hostAlloc_h(p, size, flags);
	}

	static cudaError_t cudaFreeHost(void * p)
	{
		getInstance()->free_h(p);

		return cudaSuccess;
	}

private:
	static std::mutex guard;

	cudaError_t malloc_d(void ** p, size_t size)
	{
		int device = 0;
		cudaGetDevice(&device);
		cudaError_t result = cudaSuccess;
		*p = _device.findBlock(size, device);

		if (*p == NULL)
		{
			result = ::cudaMalloc(p, size);
			if (result != cudaSuccess)
			{
				std::cout << "Memory allocation failed: " << result << "\n";
				_device.sweep();
				_array.sweep();
				result = ::cudaMalloc(p, size);
			}
			if (result == cudaSuccess)
				_device.addBlock(*p, size, device);
		}
		return result;
	}

	cudaError_t free_d(void * p)
	{
		_device.releaseBlock(p);
		return cudaSuccess;
	}

	cudaError_t malloc_array(struct cudaArray ** 	array,
		const struct cudaChannelFormatDesc * 	desc,
		size_t 	width,
		size_t 	height = 0,
		unsigned int 	flags = 0)
	{
		int device = 0;
		cudaGetDevice(&device);
		cudaError_t result = cudaSuccess;
		*array = (cudaArray*)_array.findBlock(width, height, 0, *desc, device);

		if (*array == NULL)
		{
			result = ::cudaMallocArray(array, desc, width, height, flags);
			if (result != cudaSuccess)
			{
				std::cout << "Memory allocation failed: " << result << "\n";
				_device.sweep();
				_array.sweep();
				result = ::cudaMallocArray(array, desc, width, height, flags);
			}
			if (result == cudaSuccess)
				_array.addBlock(*array, width, height, 0, *desc, device);
		}
		return result;
	}

	cudaError_t malloc_array_3d(struct cudaArray ** 	array,
		const struct cudaChannelFormatDesc * 	desc,
		const struct cudaExtent & ext,
		unsigned int 	flags = 0)
	{
		int device = 0;
		cudaGetDevice(&device);
		cudaError_t result = cudaSuccess;
		*array = (cudaArray*)_array.findBlock(ext.width, ext.height, ext.depth, *desc, device);

		if (*array == NULL)
		{
			result = ::cudaMalloc3DArray(array, desc, ext, flags);
			if (result != cudaSuccess)
			{
				std::cout << "Memory allocation failed: " << result << "\n";
				_device.sweep();
				_array.sweep();
				result = ::cudaMalloc3DArray(array, desc, ext, flags);
			}
			if (result == cudaSuccess)
				_array.addBlock(*array, ext.width, ext.height, ext.depth, *desc, device);
		}
		return result;
	}

	void free_array(void * p)
	{
		_array.releaseBlock(p);
	}

	cudaError_t malloc_h(void ** p, size_t size)
	{
		int device = 0;
		cudaGetDevice(&device);
		cudaError_t result = cudaSuccess;
		*p = _host.findBlock(size, device);

		if (*p == NULL)
		{
			result = ::cudaMallocHost(p, size);
			if (result != cudaSuccess)
			{
				std::cout << "Memory allocation failed: " << result << "\n";
				_host.sweep();
				result = ::cudaMallocHost(p, size);
			}
			if (result == cudaSuccess)
				_host.addBlock(*p, size, device);
		}
		return result;
	}

	void free_h(void * p)
	{
		if (!_host.releaseBlock(p))
		{
			_hostAlloc.releaseBlock(p);
		}
	}

	cudaError_t hostAlloc_h(void ** p, size_t size, unsigned int flags)
	{
		int device = 0;
		cudaGetDevice(&device);
		cudaError_t result = cudaSuccess;
		*p = _hostAlloc.findBlock(size, device);

		if (*p == NULL)
		{
			result = ::cudaHostAlloc(p, size, flags);
			if (result != cudaSuccess)
			{
				std::cout << "Memory allocation failed: " << result << "\n";
				_hostAlloc.sweep();
				result = ::cudaHostAlloc(p, size, flags);
			}
			if (result == cudaSuccess)
				_hostAlloc.addBlock(*p, size, device);
		}
		return result;
	}

	struct BLOCK
	{
		void * ptr;
		size_t size;
		int device;
	};

	struct ARRAY
	{
		void * ptr;
		size_t width;
		size_t height;
		size_t depth;
		cudaChannelFormatDesc desc;
		int device;
	};

	class Pool
	{
	public:
		void addBlock(void * ptr, size_t size, int device)
		{
			std::unique_lock<std::mutex> lock(_guard);

			_inUse[ptr] = { ptr, size, device };
		}

		void * findBlock(size_t size, int device)
		{
			std::unique_lock<std::mutex> lock(_guard);

			for (auto i = _free.begin(); i < _free.end(); ++i)
			{
				if (i->size == size && i->device == device)
				{
					void * p = i->ptr;
					_inUse[p] = *i;
					_free.erase(i);
					return p;
				}
			}
			return NULL;
		}

		bool releaseBlock(void * ptr)
		{
			std::unique_lock<std::mutex> lock(_guard);

			auto i = _inUse.find(ptr);

			if (i != _inUse.end())
			{
				_free.push_back(i->second);
				_inUse.erase(i);
				return true;
			}
			return false;
		}

		void sweep()
		{
			std::unique_lock<std::mutex> lock(_guard);

			for (auto i = _free.begin(); i < _free.end(); ++i)
			{
				::cudaFree(i->ptr);
			}
			_free.clear();
		}

	private:
		std::map<void*, BLOCK> _inUse;
		std::vector<BLOCK> _free;
		std::mutex _guard;
	};

	class ArrayPool
	{
	public:
		void addBlock(void * ptr, size_t width, size_t height, size_t depth, const cudaChannelFormatDesc & desc, int device)
		{
			std::unique_lock<std::mutex> lock(_guard);

			_inUse[ptr] = { ptr, width, height, depth, desc, device };
		}

		void * findBlock(size_t width, size_t height, size_t depth, const cudaChannelFormatDesc & desc, int device)
		{
			std::unique_lock<std::mutex> lock(_guard);

			for (auto i = _free.begin(); i < _free.end(); ++i)
			{
				if (i->width == width && i->height == height && i->depth == depth && i->desc.x == desc.x && i->desc.y == desc.y && i->desc.z == desc.z && i->desc.w == desc.w &&  i->desc.f == desc.f && i->device == device)
				{
					void * p = i->ptr;
					_inUse[p] = *i;
					_free.erase(i);
					return p;
				}
			}
			return NULL;
		}

		bool releaseBlock(void * ptr)
		{
			std::unique_lock<std::mutex> lock(_guard);

			auto i = _inUse.find(ptr);

			if (i != _inUse.end())
			{

				_free.push_back(i->second);

				_inUse.erase(i);

				return true;
			}
			return false;
		}

		void sweep()
		{
			std::unique_lock<std::mutex> lock(_guard);

			for (auto i = _free.begin(); i < _free.end(); ++i)
			{
				::cudaFree(i->ptr);
			}
			_free.clear();
		}

	private:
		std::map<void*, ARRAY> _inUse;
		std::vector<ARRAY> _free;
		std::mutex _guard;
	};

	Pool _device;
	Pool _host;
	Pool _hostAlloc;
	ArrayPool _array;
};

std::mutex SimpleMemoryPool::guard;

namespace
{
	static void getCurrentTimestamp()
	{
		time_t tt;
		time(&tt);
		tm * timeinfo = localtime (&tt);

		using std::chrono::system_clock;
		auto currentTime = std::chrono::system_clock::now();

		auto transformed = currentTime.time_since_epoch().count() / 1000000;
		auto millis = transformed % 1000;

		char buffer[80];
		strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M.%S", timeinfo);

		sprintf(buffer, "%s:%03d", buffer, (int)millis);
		printf("Time is %s ", buffer);	
	}

	R3DSDK::DebayerCudaJob * DebayerAllocate(const R3DSDK::AsyncDecompressJob * job, R3DSDK::ImageProcessingSettings * imageProcessingSettings, R3DSDK::VideoPixelType pixelType)
	{
		//allocate the debayer job
		R3DSDK::DebayerCudaJob *data = RED_CUDA->createDebayerJob();

		data->raw_host_mem = job->OutputBuffer;
		data->mode = job->Mode;
		data->imageProcessingSettings = imageProcessingSettings;
		data->pixelType = pixelType;

		//create raw buffer on the Cuda device
		cudaError_t err = SimpleMemoryPool::cudaMalloc(&(data->raw_device_mem), job->OutputBufferSize);

		if (err != cudaSuccess)
		{
			printf("Failed to allocate raw frame on GPU: %d\n", err);
			RED_CUDA->releaseDebayerJob(data);
			return NULL;
		}

		data->output_device_mem_size = R3DSDK::DebayerCudaJob::ResultFrameSize(data);

		//YOU MUST specify an existing buffer for the result image
		//Set DebayerCudaJob::output_device_mem_size >= result_buffer_size
		//and a pointer to the device buffer in DebayerCudaJob::output_device_mem
		err = SimpleMemoryPool::cudaMalloc(&(data->output_device_mem), data->output_device_mem_size);

		if (err != cudaSuccess)
		{
			printf("Failed to allocate result frame on card %d\n", err);
			SimpleMemoryPool::cudaFree(data->raw_device_mem);
			RED_CUDA->releaseDebayerJob(data);
			return NULL;
		}

		return data;
	}

	void DebayerFree(R3DSDK::DebayerCudaJob * job)
	{
		SimpleMemoryPool::cudaFree(job->raw_device_mem);
		SimpleMemoryPool::cudaFree(job->output_device_mem);
		RED_CUDA->releaseDebayerJob(job);
	}

	template<typename T> class ConcurrentQueue
	{
	private:
		std::mutex QUEUE_MUTEX;
		std::condition_variable QUEUE_CV;
		std::list<T *> QUEUE;

	public:
			void push(T * job)
			{
				std::unique_lock<std::mutex> lck(QUEUE_MUTEX);
				QUEUE.push_back(job);
				QUEUE_CV.notify_all();
			}

			void pop(T * & job)
			{
				std::unique_lock<std::mutex> lck(QUEUE_MUTEX);

				while (QUEUE.size() == 0)
					QUEUE_CV.wait(lck);

				job = QUEUE.front();
				QUEUE.pop_front();
			}

			size_t size() const
			{
        		return QUEUE.size();	
			}
	};

	ConcurrentQueue<R3DSDK::AsyncDecompressJob> JobQueue;

	ConcurrentQueue<R3DSDK::AsyncDecompressJob> CompletionQueue;

    void CompletionThread()
    {
		for (;;)
        {
  			R3DSDK::AsyncDecompressJob * job = NULL;

			CompletionQueue.pop(job);

			// exit thread
			if (job == NULL)
				break;

			R3DSDK::DebayerCudaJob * cudaJob = reinterpret_cast<R3DSDK::DebayerCudaJob *>(job->PrivateData);

			cudaJob->completeAsync();
				
			// frame ready for use or download etc.
			printf("Completed frame %d .\n", gpuDone);
                        
			gpuDone++;

			DebayerFree(cudaJob);

			job->PrivateData = NULL;

			// queue up next frame for decode
			if (cpuDone < TOTAL_FRAMES)
			{
				cpuDone++;
				if (GPU_DECODER->DecodeForGpuSdk(*job) != R3DSDK::DSDecodeOK)
				{
					printf("CPU decode submit failed\n");
				}
			}
        }
    }

	void GpuThread(int device)
	{
    	cudaSetDevice(device);

		cudaStream_t stream[NUM_STREAMS];

        cudaError_t err;

		for (int i = 0; i < NUM_STREAMS; ++i)
		{
			err = cudaStreamCreate(&stream[i]);
		}

		if (err != cudaSuccess)
		{
			printf("Failed to create stream %d\n", err);
			return;
		}
                
		int frameCount = 0;

		while (true)
		{
  			R3DSDK::AsyncDecompressJob * job = NULL;

			JobQueue.pop(job);

			// exit thread
			if (job == NULL)
				break;

			const R3DSDK::VideoPixelType pixelType = R3DSDK::PixelType_16Bit_RGB_Interleaved;

			R3DSDK::ImageProcessingSettings * ips = new R3DSDK::ImageProcessingSettings();
			job->Clip->GetDefaultImageProcessingSettings(*ips);

			R3DSDK::DebayerCudaJob * cudaJob = DebayerAllocate(job, ips, pixelType);

			if (err != cudaSuccess)
			{
				printf("Failed to move raw frame to card %d\n", err);
			}
                      
            int idx = frameCount++ % NUM_STREAMS;

			R3DSDK::REDCuda::Status status = RED_CUDA->processAsync(device, stream[idx], cudaJob, err);

			if (status != R3DSDK::REDCuda::Status_Ok)
			{
				printf("Failed to process frame, error %d.", status);

				if (err != cudaSuccess)
					printf(" Cuda Error: %d\n", err);
				else
					printf("\n");
			}
			else
			{
				job->PrivateData = cudaJob;
				CompletionQueue.push(job);
			}
		}

		// cleanup
        for (int i = 0; i < NUM_STREAMS; ++i)
		{
			cudaStreamDestroy(stream[i]);
		}
	}

	void CpuCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus)
	{
		JobQueue.push(item);
	}

	unsigned char * AlignedMalloc(size_t & sizeNeeded)
	{
		// alloc 15 bytes more to make sure we can align the buffer in case it isn't
		unsigned char * buffer = (unsigned char *)malloc(sizeNeeded + 15U);

		if (!buffer)
			return NULL;

		sizeNeeded = 0U;

		// cast to a 32-bit or 64-bit (depending on platform) integer so we can do the math
		uintptr_t ptr = (uintptr_t)buffer;

		// check if it's already aligned, if it is we're done
		if ((ptr % 16U) == 0U)
			return buffer;

		// calculate how many bytes we need
		sizeNeeded = 16U - (ptr % 16U);

		return buffer + sizeNeeded;
	}

	R3DSDK::REDCuda * OpenCuda(int & deviceId)
	{
		//setup Cuda for the current thread
		cudaDeviceProp deviceProp;
		cudaError_t err = cudaChooseDevice(&deviceId, &deviceProp);
		if (err != cudaSuccess)
		{
			printf("Failed to move raw frame to card %d\n", err);
			return NULL;
		}

		err = cudaSetDevice(deviceId);
		if (err != cudaSuccess)
		{
			printf("Failed to move raw frame to card %d\n", err);
			return NULL;
		}

		//SETUP YOUR CUDA API FUNCTION POINTERS
		R3DSDK::EXT_CUDA_API api;
		api.cudaFree = SimpleMemoryPool::cudaFree;
		api.cudaFreeArray = SimpleMemoryPool::cudaFreeArray;
		api.cudaFreeHost = SimpleMemoryPool::cudaFreeHost;
		api.cudaFreeMipmappedArray = ::cudaFreeMipmappedArray;
		api.cudaHostAlloc = SimpleMemoryPool::cudaHostAlloc;
		api.cudaMalloc = SimpleMemoryPool::cudaMalloc;
		api.cudaMalloc3D = ::cudaMalloc3D;
		api.cudaMalloc3DArray = SimpleMemoryPool::cudaMalloc3DArray;
		api.cudaMallocArray = SimpleMemoryPool::cudaMallocArray;
		api.cudaMallocHost = SimpleMemoryPool::cudaMallocHost;
		api.cudaMallocMipmappedArray = ::cudaMallocMipmappedArray;
		api.cudaMallocPitch = ::cudaMallocPitch;


		//CREATE THE REDCuda CLASS
		return new R3DSDK::REDCuda(api);
	}

	volatile bool firstFrameDecoded = false;

	void FirstFrameCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus)
	{
		firstFrameDecoded = true;
	}
}//end anonymous namespace

int main(int argc, char **argv)
{
	if (argc < 2)
	{
		printf("Invalid number of arguments\nExample: %s path_to_clip\n", argv[0]);
		return -1;
	}

	// initialize SDK
	R3DSDK::InitializeStatus init_status = R3DSDK::InitializeSdk(".", OPTION_RED_CUDA);
	if (init_status != R3DSDK::ISInitializeOK)
	{
		R3DSDK::FinalizeSdk();
		printf("Failed to load R3DSDK Lib: %d\n", init_status);
		return init_status;
	}

	// open CUDA device
	RED_CUDA = OpenCuda(CUDA_DEVICE_ID);

	if (RED_CUDA == NULL)
	{
		R3DSDK::FinalizeSdk();
		printf("Failed to initialize CUDA\n");
		return -1;
	}

	// load clip
	R3DSDK::Clip *clip = new R3DSDK::Clip(argv[1]);
	if (clip->Status() != R3DSDK::LSClipLoaded)
	{
		printf("Failed to load clip %d\n", clip->Status());
		delete RED_CUDA;
		cudaDeviceReset();
		return R3DSDK::DSNoClipOpen;
	}

	printf("Clip resolution = %u x %u\n", (int)clip->Width(), (int)clip->Height());

	// open CPU decoder
	GPU_DECODER = new R3DSDK::GpuDecoder();
	GPU_DECODER->Open();

	// setup threads
	std::thread * gpuThreads[GPU_THREADS];

	for (int i = 0; i < GPU_THREADS; i++)
	{
		gpuThreads[i] = new std::thread(std::bind(GpuThread, i));
	}

    std::thread completionThread(CompletionThread);

	// setup decode structure template
	const R3DSDK::VideoDecodeMode mode = R3DSDK::DECODE_FULL_RES_PREMIUM;
	const R3DSDK::VideoPixelType pixelType = R3DSDK::PixelType_16Bit_RGB_Interleaved;

	R3DSDK::AsyncDecompressJob *job = new R3DSDK::AsyncDecompressJob();
	job->Clip = clip;
	job->Mode = mode;
	job->OutputBufferSize = R3DSDK::GpuDecoder::GetSizeBufferNeeded(*job);
	size_t adjustedSize = job->OutputBufferSize;
	job->OutputBuffer = AlignedMalloc(adjustedSize);
	job->VideoFrameNo = 0;
	job->VideoTrackNo = 0;

	job->Callback = CpuCallback;

	// start printing time
	getCurrentTimestamp();

	auto start = std::chrono::system_clock::now();

	// setup decode structures for each thread and submit first FRAME_QUEUE_SIZE decodes
	size_t raw_buffer_aligned_ptr_adjustment[FRAME_QUEUE_SIZE] = { 0 };
	R3DSDK::AsyncDecompressJob *jobs[FRAME_QUEUE_SIZE] = { NULL };

	for (int i = 0; i < FRAME_QUEUE_SIZE; i++)
	{
		jobs[i] = new R3DSDK::AsyncDecompressJob();
		*jobs[i] = *job;

		raw_buffer_aligned_ptr_adjustment[i] = job->OutputBufferSize;
		jobs[i]->OutputBuffer = AlignedMalloc(raw_buffer_aligned_ptr_adjustment[i]);

		cpuDone++;

		if (GPU_DECODER->DecodeForGpuSdk(*(jobs[i])) != R3DSDK::DSDecodeOK)
		{
			printf("CPU decode submit failed\n");
			return R3DSDK::DSDecodeFailed;
		}
	}

	// wait for work to complete
	while (gpuDone < TOTAL_FRAMES)
	{
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

	// end current time

	getCurrentTimestamp();

	auto end = std::chrono::system_clock::now();

	auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	
	float fps = 1000.0f / ((float)diff / (float)TOTAL_FRAMES);

	std::cout << "\n" << diff << " = " << fps << " fps\n";

	// instruct threads to exit
	{
		for (int i = 0; i < GPU_THREADS; i++)
			JobQueue.push(NULL);
	}

	// wait for threads to exit
	for (int i = 0; i < GPU_THREADS; i++)
	{
		gpuThreads[i]->join();
		delete gpuThreads[i];
	}

    CompletionQueue.push(NULL);

	completionThread.join();

	// cleanup
	for (int i = 0; i < FRAME_QUEUE_SIZE; i++)
	{
		free(((unsigned char *)jobs[i]->OutputBuffer) - raw_buffer_aligned_ptr_adjustment[i]);
		delete jobs[i];
	}

	delete job;

	delete GPU_DECODER;
	delete clip;
	delete RED_CUDA;

	cudaDeviceReset();
	R3DSDK::FinalizeSdk();
	return 0;
}
