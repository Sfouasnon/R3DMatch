/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

#include <stddef.h>
#include <iostream>
#include <Metal/Metal.h>
#include <CoreServices/CoreServices.h>
#include <iostream>
#include <fstream>
#include <queue>
#include <thread>
#include <algorithm>
#include <sstream>

#include "R3DSDKDefinitions.h"
#include "R3DSDK.h"
#include "R3DSDKMetal.h"

namespace
{
    template<typename T> class concurrent_queue
    {
    public:
        void push(const T & v)
        {
            std::lock_guard<std::mutex> l(g_);
            
            q_.push(v);
        }
        
        bool pop(T & v)
        {
            std::lock_guard<std::mutex> l(g_);
            
            if (q_.empty()) return false;
            
            v = q_.front();
            q_.pop();
            
            return true;
        }
        
        void clear()
        {
            std::lock_guard<std::mutex> l(g_);
            
            while(!q_.empty())
            {
                q_.pop();
            }
        }
        
        size_t size() const
        {
            return q_.size();
        }
        
    private:
        std::queue<T> q_;
        std::mutex g_;
    };

    std::vector < R3DSDK::VideoDecodeMode > DecodeModes()
	{
		std::vector < R3DSDK::VideoDecodeMode > modes;
		modes.push_back(R3DSDK::DECODE_FULL_RES_PREMIUM);
		//modes.push_back(R3DSDK::DECODE_HALF_RES_PREMIUM);
		//modes.push_back(R3DSDK::DECODE_HALF_RES_GOOD);
		//modes.push_back(R3DSDK::DECODE_QUARTER_RES_GOOD);
		//modes.push_back(R3DSDK::DECODE_EIGHT_RES_GOOD);
		//modes.push_back(R3DSDK::DECODE_SIXTEENTH_RES_GOOD);
		return modes;
	}

	class MetalTestContext
	{
	public:
		MetalTestContext() : device()
		{
            NSArray<id<MTLDevice>> * devices = MTLCopyAllDevices();
            
            for ( int i = 0; i < devices.count; ++i )
            {
                NSLog(@"%@", devices[i].description);
                
            }
            
			device = MTLCreateSystemDefaultDevice();
            
            NSLog(@"Selected %@", device.description);
		}

		~MetalTestContext()
		{
			 
		}

		id<MTLDevice> device;
	};

	class MetalTestQueue
	{
	public:
		MetalTestQueue(id<MTLDevice> device) : queue()
		{
			queue = [device newCommandQueue];
		}

		~MetalTestQueue()
		{
		}

		id<MTLCommandQueue> queue;
	};

	class MetalTestPlatform
	{
	public:
		MetalTestPlatform() : device(), queue()
		{

			device = std::make_shared<MetalTestContext>();

			queue = std::make_shared<MetalTestQueue>(device->device);
		}
	public:
		std::shared_ptr<MetalTestContext> device;
		std::shared_ptr<MetalTestQueue> queue;
	};
	
	class ImageData
	{
	public:
        ImageData(std::shared_ptr<R3DSDK::ImageProcessingSettings> image_processing_settings, size_t buffer_size, std::shared_ptr<char> raw_buffer, R3DSDK::VideoDecodeMode decode_mode,int clip_width, int clip_height, std::string clip_id) :
			ips(image_processing_settings)
			, rawBuffer(raw_buffer)
			, bufferSize(buffer_size)
			, mode(decode_mode)
			, width(clip_width)
			, height(clip_height)
            , clipId(clip_id)
		{
		}

		~ImageData()
		{
			ips = std::shared_ptr<R3DSDK::ImageProcessingSettings>();
			rawBuffer = std::shared_ptr<char>();
		}
		std::shared_ptr<R3DSDK::ImageProcessingSettings> ips;
		std::shared_ptr<char> rawBuffer;
		size_t bufferSize;
		R3DSDK::VideoDecodeMode mode;
		int width;
		int height;
        std::string clipId;
	};
	
	volatile bool decodeDone = false;
	void asyncCallback(R3DSDK::AsyncDecompressJob * item, R3DSDK::DecodeStatus decodeStatus)
	{
        if (!item)
        {
            return;
        }
		//static size_t idx = 0;
		//printf("DecodeForGPUSDK: %d Idx: %lu %C%C%C%C Sensor: %s\n", decodeStatus, idx, (item->Mode >> 24) & 0xFF, (item->Mode >> 16) & 0xFF, (item->Mode >> 8) & 0xFF, (item->Mode) & 0xFF, item->Clip->MetadataItemAsString(R3DSDK::RMD_SENSOR_NAME).c_str());
		//idx++;
		item->PrivateData = (void *)decodeStatus;
		decodeDone = true;
	}

	class REDMetalTestEnvironment {
	public:
		virtual R3DSDK::InitializeStatus SetUp() {

            sPlatform = std::make_shared<MetalTestPlatform>();

            R3DSDK::InitializeStatus status =  R3DSDK::InitializeSdk("./", OPTION_RED_METAL);

			if( status != R3DSDK::ISInitializeOK )
            {
                sPlatform = std::shared_ptr<MetalTestPlatform>();
                printf("Failed to initalize R3DSDK: %d\n", status);
            }
            return status;
        }
        
        virtual void LoadClip(const char * path) {
            
			printf("SDK Version: %s\n", R3DSDK::GetSdkVersion());
			sAsyncDecoder = std::make_shared<R3DSDK::GpuDecoder>();
			sAsyncDecoder->Open();
			std::shared_ptr<R3DSDK::AsyncDecompressJob> job = std::make_shared<R3DSDK::AsyncDecompressJob>();

			std::vector<R3DSDK::VideoDecodeMode> modes = DecodeModes();
			int idx = 0;

            std::shared_ptr<R3DSDK::Clip> clip = std::make_shared<R3DSDK::Clip>(path);
            if (clip->Status() != R3DSDK::LSClipLoaded)
            {
                printf("Failed to load: %s %d\n", path, clip->Status());
                return;
            }

            for (std::vector<R3DSDK::VideoDecodeMode>::iterator mode_it = modes.begin(); mode_it != modes.end(); ++mode_it)
            {
                job->Clip = clip.get();
                job->Mode = *mode_it;
                
                int frameCount = (int)clip->VideoFrameCount();

                for (int frameNo = 0; frameNo < frameCount; ++frameNo)
                {
                    size_t bufferSizeNeeded = R3DSDK::GpuDecoder::GetSizeBufferNeeded(*(job.get()));
                    std::shared_ptr<char> buffer(new char[bufferSizeNeeded], std::default_delete<char>());
                    job->OutputBuffer = buffer.get();
                    job->OutputBufferSize = bufferSizeNeeded;
                    job->OutputFrameMetadata = NULL;
                    job->PrivateData = NULL;
                    job->VideoFrameNo = frameNo;
                    job->VideoTrackNo = 0;
                    job->AbortDecode = false;
                    job->Callback = asyncCallback;

                    decodeDone = false;

                    R3DSDK::DecodeStatus ds = sAsyncDecoder->DecodeForGpuSdk(*(job.get()));
                    printf("Decode status: %d\n", ds);
                    if (ds != R3DSDK::DSDecodeOK)
                    {
                        printf("Failed to DecodeForGPUSDK: %s at mode %c%c%c%c %d\n", path, ((*mode_it) >> 24) & 0xFF, ((*mode_it) >> 16) & 0xFF, ((*mode_it) >> 8) & 0xFF, (*mode_it) & 0xFF, clip->Status());
                        return;
                    }
                    
                    while (!decodeDone)
                    {
                    }

                    //decoded successfully add to image data list.
                    std::shared_ptr<R3DSDK::ImageProcessingSettings> ips = std::make_shared<R3DSDK::ImageProcessingSettings>();
                    clip->GetDefaultImageProcessingSettings(*(ips.get()));
                    std::shared_ptr<ImageData> data = std::make_shared<ImageData>(ips, bufferSizeNeeded, buffer, *mode_it, clip->Width(), clip->Height(), clip->MetadataItemAsString(R3DSDK::RMD_ORIGINAL_FILENAME));
                    sImageData.push(data);

                    printf("Idx: %d %c%c%c%c clip resolution = %d x %d\n", idx, (*mode_it >> 24) & 0xFF, (*mode_it >> 16) & 0xFF, (*mode_it >> 8) & 0xFF, (*mode_it) & 0xFF, (int)clip->Width(), (int)clip->Height());
                    ++idx;
                }
            }
		}

		virtual void TearDown() 
		{
			sImageData.clear();
			if (sAsyncDecoder ) sAsyncDecoder->Close();
			sAsyncDecoder = std::shared_ptr<R3DSDK::GpuDecoder>();
			R3DSDK::FinalizeSdk();
		}
	public:
		static std::shared_ptr<R3DSDK::GpuDecoder> sAsyncDecoder;
		static concurrent_queue< std::shared_ptr<ImageData> > sImageData;

		static std::shared_ptr<MetalTestPlatform> sPlatform;
	};
	std::shared_ptr<R3DSDK::GpuDecoder> REDMetalTestEnvironment::sAsyncDecoder;
	concurrent_queue< std::shared_ptr<ImageData> > REDMetalTestEnvironment::sImageData;
	std::shared_ptr<MetalTestPlatform> REDMetalTestEnvironment::sPlatform;
}

@interface BUFFER : NSObject
@property size_t size;
@property MTLResourceOptions options;
@property id<MTLBuffer> buffer;
@property (readonly) NSString *inusekey;
@property (readonly) NSString *freekey;

+(NSString*)inUseKey:(id<MTLBuffer>) buffer;
+(NSString*)freeKey:(size_t) size options:(MTLResourceOptions)options;
-(id)initWithBuffer: (id<MTLBuffer>) buffer size:(size_t)size options:(MTLResourceOptions)options;
@end

@implementation BUFFER
+(NSString*)inUseKey:(id<MTLBuffer>) buffer{
    return [NSString stringWithFormat:@"%p", buffer];
}
+(NSString*)freeKey:(size_t) size options:(MTLResourceOptions)options{
    return [NSString stringWithFormat:@"%lu:%lu", size, (unsigned long)options];
}

-(id)initWithBuffer: (id<MTLBuffer>) buffer size:(size_t)size options:(MTLResourceOptions)options
{
    self = [super init];
    
    if (self)
    {
        _size = size;
        _buffer = buffer;
        _options = options;
        _freekey = [BUFFER freeKey:size options:options];
        _inusekey = [BUFFER inUseKey:buffer];
    }
    return self;
}
@end

@interface TEXTURE : NSObject
@property id<MTLTexture> texture;
@property (readonly) NSString *inusekey;
@property (readonly) NSString *freekey;

+(NSString*)inUseKey:(id<MTLTexture>) texture;
+(NSString*)freeKey:(MTLTextureDescriptor*) descriptor;

-(id)initWithTexture: (id<MTLTexture>) texture descriptor:(MTLTextureDescriptor *)descriptor;
@end

@implementation TEXTURE
+(NSString*)inUseKey:(id<MTLTexture>) texture{
    return [NSString stringWithFormat:@"%p", texture];
}
+(NSString*)freeKey:(MTLTextureDescriptor*) descriptor{
    return [NSString stringWithFormat:@"%d:%d:%d:%d:%d:%d",
            (int)descriptor.textureType,
            (int)descriptor.pixelFormat,
            (int)descriptor.width,
            (int)descriptor.height,
            (int)descriptor.depth,
            (int)descriptor.usage];
}

-(id)initWithTexture: (id<MTLTexture>) texture descriptor:(MTLTextureDescriptor *)descriptor
{
    self = [super init];
    
    if (self)
    {
        _texture = texture;
        _freekey = [TEXTURE freeKey:descriptor];
        _inusekey = [TEXTURE inUseKey:texture];
    }
    return self;
}
@end

NSMutableDictionary<NSString *, BUFFER *> *buffers;
NSMutableDictionary<NSString *, TEXTURE *> *textures;

NSMutableDictionary<NSString *, NSMutableArray<BUFFER*> *> *freeBuffers;
NSMutableDictionary<NSString *, NSMutableArray<TEXTURE*> *> *freeTextures;

std::mutex bufferMutex;
std::mutex textureMutex;

void RunSample();

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        printf("Usage: MetalGpuDecodeSample file.R3D\n");
        return -1;
    }
    
    buffers = [[NSMutableDictionary alloc] init];
    freeBuffers = [[NSMutableDictionary alloc] init];
    
    textures = [[NSMutableDictionary alloc] init];
    freeTextures = [[NSMutableDictionary alloc] init];

	REDMetalTestEnvironment env;
    
    R3DSDK::InitializeStatus status = env.SetUp();
    
    if (status != R3DSDK::ISInitializeOK)
    {
        return status;
    }
    
    std::thread th(RunSample);

	env.LoadClip(argv[1]);
    
    REDMetalTestEnvironment::sImageData.push(NULL);

    th.join();

	env.TearDown();
	
    return 0;
}

id<MTLBuffer> createMTLBuffer(id<MTLDevice> device, size_t size, void * hostPtr, MTLResourceOptions mode, int &error)
{
    std::lock_guard<std::mutex> lock(bufferMutex);

    id<MTLBuffer> buffer = nil;
    
    NSString * key = [BUFFER freeKey:size options:mode];
    
    NSMutableArray<BUFFER*> * list = freeBuffers[key];
    
    BUFFER * block;
    
    if (list && list.count > 0)
    {
        block = [list lastObject];
        [list removeLastObject];
        buffers[block.inusekey] = block;
        buffer = block.buffer;
        
        if (hostPtr && mode != MTLResourceStorageModePrivate)
        {
            memcpy(buffer.contents, hostPtr, size);
            [buffer didModifyRange:NSMakeRange(0, size)];
        }
    }
    else
    {
        if (hostPtr && mode != MTLResourceStorageModePrivate)
        {
            buffer = [device newBufferWithBytes:hostPtr length:size options:mode];
            [buffer didModifyRange:NSMakeRange(0, size)];
        }
        else
        {
            buffer = [device newBufferWithLength:size options:mode];
        }
        if (buffer == nil)
        {
            error = -1;
        }
        block = [[BUFFER alloc] initWithBuffer:buffer size:size options: mode];
        buffers[block.inusekey] = block;
    }
    return buffer;
}

id<MTLTexture> createMTLTexture(id<MTLDevice> device, MTLTextureDescriptor * descriptor, int &error)
{
    id<MTLTexture> texture = nil;
    
    NSString * key = [TEXTURE freeKey:descriptor];

    {
        std::lock_guard<std::mutex> lock(textureMutex);

        TEXTURE *block;
        
        NSMutableArray<TEXTURE*> * list = freeTextures[key];
    
        if (list && list.count > 0)
        {
            block = [list lastObject];
            [list removeLastObject];
            textures[block.inusekey] = block;
            texture = block.texture;
        }
        else
        {
            texture = [device newTextureWithDescriptor:descriptor];
        
            block = [[TEXTURE alloc] initWithTexture: texture descriptor: descriptor];
            textures[block.inusekey] = block;
        }
    }

    return texture;
}

void releaseMTLBuffer(id<MTLBuffer> buffer)
{
    std::lock_guard<std::mutex> lock(bufferMutex);

    BUFFER * block = buffers[[BUFFER inUseKey:buffer]];
    
    if (block)
    {
        NSMutableArray<BUFFER*> * list = freeBuffers[block.freekey];
        if (!list)
        {
            list = [[NSMutableArray<BUFFER*> alloc] init];
            freeBuffers[block.freekey] = list;
        }
        [list addObject:block];

        [buffers removeObjectForKey:block.inusekey];
    }
}

void releaseMTLTexture(id<MTLTexture> texture)
{
    std::lock_guard<std::mutex> lock(textureMutex);

    TEXTURE * block = textures[[TEXTURE inUseKey:texture]];
    
    if (block)
    {
        NSMutableArray<TEXTURE*> * list = freeTextures[block.freekey];
        if (!list)
        {
            list = [[NSMutableArray<TEXTURE*> alloc] init];
            freeTextures[block.freekey] = list;
        }
        [list addObject:block];
        
        [textures removeObjectForKey:block.inusekey];
    }
}

void completionThreadRun(R3DSDK::REDMetal * red_mtl, concurrent_queue<R3DSDK::DebayerMetalJob *> *completeQueue, std::atomic<int> *jobCount)
{
    int frameCount = 0;
    for (;;)
    {
        R3DSDK::DebayerMetalJob * job;
        
        while (!completeQueue->pop(job))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!job) break;
        job->completeAsync();
        
        printf("Completed frame %d\n", ++frameCount);
        
        if (false)
        {
            std::stringstream ss;
            ss << "Frame_" << frameCount << ".raw";
            FILE * fp = fopen(ss.str().c_str(), "wb+");
            id<MTLBuffer> buffer = job->output_device_mem;
            fwrite([buffer contents], 1, [buffer length], fp);
            fclose(fp);
        
        }
        
        releaseMTLBuffer(job->raw_device_mem);
        if (job->output_device_mem != nil) releaseMTLBuffer(job->output_device_mem);
        if (job->output_device_image != nil) releaseMTLTexture(job->output_device_image);
        
        red_mtl->releaseDebayerJob(job);
        --(*jobCount);
    }
}

void RunSample()
{
	if (!REDMetalTestEnvironment::sPlatform)
	{
		printf("Skipped Decode Test, no Metal available\n");
		return;
	}

    R3DSDK::EXT_METAL_API metal_api;
    
    metal_api.createMTLBuffer = createMTLBuffer;
    metal_api.createMTLTexture = createMTLTexture;
    metal_api.releaseMTLBuffer = releaseMTLBuffer;
    metal_api.releaseMTLTexture = releaseMTLTexture;

    R3DSDK::REDMetal * redmtl = new R3DSDK::REDMetal(metal_api);
    
#define BATCH_SIZE 4
    
    id<MTLCommandQueue> commandQueue = REDMetalTestEnvironment::sPlatform->queue->queue;
    
    concurrent_queue<R3DSDK::DebayerMetalJob *> completeQueue;
    
    std::atomic<int> jobCount(0);
    
    std::thread completeThread(std::bind(completionThreadRun, redmtl, &completeQueue, &jobCount));
    
    int err = 0;
    
    for (int imgIdx = 0; true; ++imgIdx)
    {
        @autoreleasepool {

            std::shared_ptr<ImageData> imgData;
            
            bool result = REDMetalTestEnvironment::sImageData.pop(imgData);
            
            if (!result)
            {
                continue;
            }
            
            if (imgData == NULL)
            {
                break;
            }
            
            id<MTLBuffer> rawDeviceBuffer = createMTLBuffer(REDMetalTestEnvironment::sPlatform->device->device, imgData->bufferSize, NULL, MTLResourceStorageModeManaged, err);

            R3DSDK::DebayerMetalJob * job = redmtl->createDebayerJob();
            
            job->imageProcessingSettings = imgData->ips.get();
            job->mode = imgData->mode;
            job->raw_host_mem = imgData->rawBuffer.get();
            job->pixelType = R3DSDK::PixelType_16Bit_RGB_Interleaved;
            
           
//#define USE_MTL_TEXTURE
            
#if !defined(USE_MTL_TEXTURE)
            //
            // Okay to call ResultFrameSize here if we are using output_device_mem
            //
            size_t resultSize = R3DSDK::DebayerMetalJob::ResultFrameSize(*(job));

            id<MTLBuffer> resultDeviceBuffer = createMTLBuffer(REDMetalTestEnvironment::sPlatform->device->device, resultSize, NULL, MTLResourceStorageModeManaged, err);
            
            job->output_device_mem = resultDeviceBuffer;
#else
            MTLTextureDescriptor * descriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Uint width:imgData->width height:imgData->height mipmapped:false];
            descriptor.usage = MTLTextureUsageShaderWrite;
            id<MTLTexture> resultDeviceImage = createMTLTexture(REDMetalTestEnvironment::sPlatform->device->device, descriptor, err);
            job->output_device_image = resultDeviceImage;
            //
            // We need to call ResultFrameSize here, after setting output_device_image to get the correct size
            //
            size_t resultSize = R3DSDK::DebayerMetalJob::ResultFrameSize(*(job));
#endif
            
            job->output_device_mem_size = resultSize;
            job->raw_device_mem = rawDeviceBuffer;
            job->batchMode = true;
            
            R3DSDK::REDMetal::Status status = redmtl->processAsync(commandQueue, job, err);
            
            id<MTLCommandBuffer> cmdBuffer = [commandQueue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmdBuffer blitCommandEncoder];
            [blit synchronizeResource:resultDeviceBuffer];
            [blit endEncoding];
            [cmdBuffer commit];
            
            if (status != R3DSDK::REDMetal::Status_Ok)
            {
                std::cout << "Failed to process frame " << imgIdx << ". Status: " << status << std::endl;
                break;
            }
            
            completeQueue.push(job);
            
            ++jobCount;

            if ((imgIdx + 1) % BATCH_SIZE == 0)
            {
                redmtl->flush(commandQueue, err);
                
                while ( jobCount >= BATCH_SIZE * 2 )
                {
                    std::this_thread::yield();
                }
            }
        }
    }
    
    redmtl->flush(commandQueue, err);
    
    completeQueue.push(NULL);
    
    completeThread.join();
    
    delete redmtl;
}
