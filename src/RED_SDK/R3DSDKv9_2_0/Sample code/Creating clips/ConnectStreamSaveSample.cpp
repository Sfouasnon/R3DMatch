/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* Sample to demonstrate saving the RED Connect stream to an
   R3D file, using the R3DSDK's I/O back-end with a custom
   implementation. This can be used to store the RED Connect
   stream in the cloud or other storage systems, provided the
   media can keep up with the data rate from the camera */

#ifndef _WIN32
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <poll.h>
#  include <sys/types.h>
#  include <sys/stat.h>
#  include <unistd.h>
#else
#  include <ws2tcpip.h>
#  include <mstcpip.h>
#  include <direct.h>
#  include <sys/stat.h>
#  define mkdir(dir, mode) _mkdir(dir)
#  pragma comment(lib, "Ws2_32")
#endif

#include <iostream>
#include <string>
#include <vector>

#include <R3DSDK.h>
#include <R3DSDKCustomIO.h>
#include <R3DSDKStream.h>


#ifndef _WIN32
#define sprintf_s(buf, size, ...) snprintf((buf), (size), __VA_ARGS__)
#define fprintf_s fprintf
typedef int errno_t;
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))

#define strerror_s(buf,len,errno) strerror_r(errno,buf,len) 
#define MAX_PATH 1024

#include <string.h>
#include <unistd.h>

#define __int64 __int64_t
#define _int64 __int64_t

#define strcpy_s(dest, dstSize, src) strncpy((dest), (src), (dstSize))
#define strcat_s(dest, dstSize, src) strncat((dest), (src), (dstSize))

#define memmove_s(dest, dsize, src, count) memmove((dest), (src), (count))

#define sscanf_s sscanf

#define stricmp strcasecmp
#define _strnicmp strncasecmp

typedef int SOCKET;
typedef unsigned int DWORD;

#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)

int WSAGetLastError() 
{
    return errno;
}
#endif

#ifndef min
#  define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif
#ifndef max
#  define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef mode_t
#  define mode_t unsigned int
#endif
typedef struct stat Stat;

static bool exitApp = false;

static bool getExitApp()
{
    return exitApp;
}


#ifdef _WIN32
static void setExitApp()
{
    std::cout << "Preparing to exit." << std::endl;
    exitApp = true;
}

// Code to trap Windows Console Control-C to allow for a graceful exit from service mode
static BOOL WINAPI consoleHandler(DWORD signal)
{
    if (signal == CTRL_C_EVENT)
        setExitApp();
    return TRUE;
}
#endif

//*************************************************
// R3DSDK Redirected IO code
//
// This code intercepts R3DSDK file access calls
//
// These routines can be used to redirect the files to 
// non-standard storage devices or locations
//*************************************************
#define USE_CUSTOM_IO (1)

#ifdef USE_CUSTOM_IO
class IOSample : public R3DSDK::IOInterface
{
public:
    virtual Handle Open(const char * utf8Path, FileAccess access)
    {
        if (access == IO_READ)
            return HANDLE_FALLBACK;

        //printf("Custom I/O creating file: %s\n", utf8Path);

        FILE * f;
        fopen_s(&f, utf8Path, "wb");

        if (f == NULL)
            return R3DSDK::IOInterface::HANDLE_ERROR;

        return reinterpret_cast<Handle>(f);
    }

    virtual void Close(Handle handle)
    {
        fclose(reinterpret_cast<FILE *>(handle));
    }

    virtual unsigned long long Filesize(Handle handle)
    {
        return 0;
    }

    virtual bool Read(void * outBuffer, size_t bytes, unsigned long long offset, Handle handle)
    {
        return false;
    }

    virtual bool Write(const void * inBuffer, size_t bytes, Handle handle)
    {
        return (fwrite(inBuffer, bytes, 1, reinterpret_cast<FILE *>(handle)) == 1);
    }

    virtual bool CreatePath(const char * utf8Path)
    {
        std::string directory = (char *)utf8Path;
        if (directory.back() == ':')
            return true;
        if (directory.back() != '/' && directory.back() != '\\')
#ifdef _WIN32
            directory += '\\';
#else
			directory += '/';
#endif
        return 0 == mkpath(directory.c_str(), 0777);
    }

private:
    int do_mkdir(const char *path, mode_t mode)
    {
        Stat            st;
        int             status = 0;

        if (stat(path, &st) != 0)
        {
            /* Directory does not exist. EEXIST for race condition */
            if (mkdir(path, mode) != 0 && errno != EEXIST)
                status = -1;
        }
        else if (!(st.st_mode & S_IFDIR))
        {
            errno = ENOTDIR;
            status = -1;
        }

        return(status);
    }


    /**
    ** mkpath - ensure all directories in path exist
    ** Algorithm takes the pessimistic view and works top-down to ensure
    ** each directory in path exists, rather than optimistically creating
    ** the last element and working backwards.
    */
    int mkpath(const char *path, mode_t mode)
    {
        char			*pp;
        char			*sp = 0;
        char			*sprev = 0;
        int				status;
        char			copypath[1024];

        strcpy_s(copypath, sizeof(copypath) - 1, path);

        status = 0;
        pp = copypath;

        while ((status == 0) && (((sp = strchr(pp, '/')) != 0) || (sprev = strchr(pp, '\\')) != 0))
        {
            if (!sp)
                sp = sprev;
            char dt = *sp;
            if (sp != pp)
            {
                /* Neither root nor double slash in path */
                *sp = '\0';
                status = do_mkdir(copypath, mode);
                *sp = dt;
            }
            pp = sp + 1;
            sp = 0;
            sprev = 0;
        }
        if (status == 0)
            status = do_mkdir(path, mode);
        return (status);
    }
};
#endif // USE_CUSTOM_IO

//*************************************************
// Support routines for logging TCP information
//*************************************************
static std::string getErrorText(int sts)
{
#ifdef _WIN32
    char *errText = NULL;
    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, sts,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&errText, 0, NULL);
    for (size_t i = strlen(errText) - 1; i >= 0 && (errText[i] == '\r' || errText[i] == '\n'); i--)
        errText[i] = 0;
    std::string errorString = errText;
    if (errText)
        LocalFree(errText);
    return errorString;
#else
    return strerror(sts);
#endif
}

static bool addPollfdEventDesc(std::string &eventDescs, short eventMask, short events, const char *desc)
{
    if (events == (eventMask & events))
    {
        if (eventDescs.length())
            eventDescs += ", ";
        eventDescs += desc;
        return true;
    }
    return false;
}

static std::string getPollfdEventDesc(short eventMask)
{
    std::string eventDesc("");
    if (!addPollfdEventDesc(eventDesc, eventMask, POLLIN, "POLLIN"))	// POLLIN is a combination of POLLRDNORM and POLLRDBAND, so if POLLIN is found, don't look for the components
    {
        addPollfdEventDesc(eventDesc, eventMask, POLLRDNORM, "POLLRDNORM");
        addPollfdEventDesc(eventDesc, eventMask, POLLRDBAND, "POLLRDBAND");
    }
    addPollfdEventDesc(eventDesc, eventMask, POLLPRI, "POLLPRI");

    addPollfdEventDesc(eventDesc, eventMask, POLLOUT, "POLLOUT(POLLWRNORM)");
    addPollfdEventDesc(eventDesc, eventMask, POLLWRBAND, "POLLWRBAND");

    addPollfdEventDesc(eventDesc, eventMask, POLLERR, "POLLERR");
    addPollfdEventDesc(eventDesc, eventMask, POLLHUP, "POLLHUP");
    addPollfdEventDesc(eventDesc, eventMask, POLLNVAL, "POLLNVAL");

    return eventDesc;
}

//*************************************************
//*************************************************
//*************************************************
// This is the ReceiveStream class
//
// This class will read the RED Connect R3D stream 
// and call the R3DSDK to save the stream to an R3D
// file.
//
// The R3DSDK's redirected IO functionality is being 
// used to save the files to the local disk.  The 
// redirected IO routines can be used to save the 
// R3D stream to R3D files on non-standard storage 
// devices or locations
//*************************************************
//*************************************************
//*************************************************

class ReceiveStream
{
public:
    ReceiveStream()
        : mExiting(false)
        , mKeepGoing(true)
        , mHost("")
        , mPort("9000")
        , mFolder(".\\")
        , mR3DSDKFolder(".")
        , mClipId(1)
        , mReelId(1)
        , mKeepAlive(false)
        , mHandle(0)
        , mTcpClientSocket(INVALID_SOCKET)
        , mTcpKeepAliveEnabled(false)
    {
    }

    /////////////////////////
    // Start initializes and starts the Receive Loop
    /////////////////////////
    int Start()
    {
        mExiting  = false;

		R3DSDK::InitializeStatus sts = R3DSDK::InitializeSdk(mR3DSDKFolder.c_str(), 0);
		
        std::cout << R3DSDK::GetSdkVersion() << std::endl;
        
		if (sts != R3DSDK::ISInitializeOK)
        {
            std::cout << "Failed to initialize SDK " << mR3DSDKFolder << ", Error: " << (int)sts << std::endl;
            return 3;
        }

#ifdef USE_CUSTOM_IO
        IOSample * sample = new IOSample();
        R3DSDK::SetIoInterface(sample);
#endif

#if defined(_WIN32)
        WSADATA wsaData;        //Winsock startup info
        int error;
        if (error = WSAStartup(MAKEWORD(2, 2), &wsaData))
        { // there was an error
            std::wcout << "Unable to initialize Windows Sockets." << std::endl;
            return -1;
        }
        if (wsaData.wVersion != MAKEWORD(2, 2))
        { // wrong WinSock version!
            std::cout << "Invalid Windows Sockets version.  Winsock V2 is required for the TCP Receive Protocol." << std::endl;
            WSACleanup(); // unload ws2_32.dll
            return -2;
        }
#endif

        mTcpClientSocket = 0;

        if ((mTcpClientSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)) < 0)
        {
            std::cout << "***** Error setting up TCP Receive Protocol socket creation failed. " << mHost << std::endl;
            return -3;
        }

        mTcpKeepAliveEnabled = false;

        /* Set the option active */
        int optval = 1;
        socklen_t optlen = sizeof(optval);
        if (setsockopt(mTcpClientSocket, SOL_SOCKET, SO_OOBINLINE, (char *)(char *)&optval, optlen) < 0)
        {
            int socketErr = WSAGetLastError();
            std::cout << "TCP Receive Protocol setsocket failed while attempting to enable Out of Band Inline. Warning " << socketErr << ". " << getErrorText(socketErr) << std::endl;
        }

        // Allow large transfers at 1500 MTU
        int bufferSize = 8 * 1024 * 1024;        if (setsockopt(mTcpClientSocket, SOL_SOCKET, SO_RCVBUF, (char*)(&bufferSize), sizeof(bufferSize)) < 0)        {            int socketErr = WSAGetLastError();
            std::cout << "TCP Receive Protocol setsocket failed while attempting to set SO_RCVBUF: " << socketErr << ". " << getErrorText(socketErr) << std::endl;        }        if (setsockopt(mTcpClientSocket, SOL_SOCKET, SO_SNDBUF, (char*)(&bufferSize), sizeof(bufferSize)) < 0)        {            int socketErr = WSAGetLastError();
            std::cout << "TCP Receive Protocol setsocket failed while attempting to set SO_SNDBUF: " << socketErr << ". " << getErrorText(socketErr) << std::endl;        }


#ifdef _WIN32
        DWORD dwBytesRet = 0;
        struct tcp_keepalive alive;
        alive.onoff = TRUE;
        alive.keepalivetime = 1000;         // Wait for 1000 ms of no activity before sending the first keep alive request
        alive.keepaliveinterval = 1000;     // If still no activity, continue sending keep alive requests every second.

        if (WSAIoctl(mTcpClientSocket, SIO_KEEPALIVE_VALS, &alive, sizeof(alive), NULL, 0, &dwBytesRet, NULL, NULL) == SOCKET_ERROR)
        {
            int socketErr = WSAGetLastError();
            std::cout << "TCP Receive Protocol WSAIoctl failed to set the keep alive settings. Warning " << socketErr << ". " << getErrorText(socketErr) << std::endl;
        }
        else
        {
            std::cout << "TCP Receive Protocol successfully set the keep alive settings. keep alive is " << (alive.onoff ? "On, " : "Off, ") << alive.keepalivetime << " ms to first keep alive request, " << alive.keepaliveinterval << " ms between keep alive requests" << std::endl;

            // This might be redundant (based on the previous SIO_KEEPALIVEVALS WSAIoctl call)
            optlen = sizeof(optval);
            /* Check the status for the keepalive option */
            if (getsockopt(mTcpClientSocket, SOL_SOCKET, SO_KEEPALIVE, (char *)&optval, &optlen) < 0)
            {
                int socketErr = WSAGetLastError();
                std::cout << "TCP Receive Protocol getsocket failed to get keep alive status. Warning " << socketErr << ". " << getErrorText(socketErr) << std::endl;
            }
            else if (!optval)
            {
                /* Set the option active */
                optval = 1;
                optlen = sizeof(optval);
                if (setsockopt(mTcpClientSocket, SOL_SOCKET, SO_KEEPALIVE, (char *)(char *)&optval, optlen) < 0)
                {
                    int socketErr = WSAGetLastError();
                    std::cout << "TCP Receive Protocol setsocket failed while attempting to enable keep alive. Warning " << socketErr << ". " << getErrorText(socketErr) << std::endl;
                }
                else
                {
                    /* Check the status again */
                    if (getsockopt(mTcpClientSocket, SOL_SOCKET, SO_KEEPALIVE, (char *)&optval, &optlen) < 0)
                    {
                        int socketErr = WSAGetLastError();
                        std::cout << "TCP Receive Protocol setsocket failed to enable keep alive. Warning " << socketErr << ". " << getErrorText(socketErr) << std::endl;
                    }
                    else
                    {
                        std::cout << "TCP Receive Protocol setsocket successfully enabled keep alive." << std::endl;
                        mTcpKeepAliveEnabled = true;
                    }
                }
            }
        }
#endif

        uint16_t port = (uint16_t)std::stoul(mPort);
        struct sockaddr_in serv_addr;

        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port   = htons(port);

        bool exitingRoutine = false;

        // Convert IPv4 addresses from text to binary form
        if (!exitingRoutine && (inet_pton(AF_INET, mHost.c_str(), &serv_addr.sin_addr) <= 0))
        {
            int socketErr = WSAGetLastError();
            std::cout << "TCP Receive Protocol camera address decode failed. Error " << socketErr << ". " << getErrorText(socketErr) << std::endl;
            exitingRoutine = true;
        }

        static bool sucessfullyConnected = false;

        if (!exitingRoutine && (connect(mTcpClientSocket, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0))
        {
            int socketErr = WSAGetLastError();
            if (sucessfullyConnected)           // Don't repeatedly log this error, just log the first time it happens, or the first time it happens after a successfull connection
            {
                std::cout << "TCP Receive Protocol Connection failed.  Error " << socketErr << ". " << getErrorText(socketErr) << std::endl;
            }
            sucessfullyConnected = false;
            exitingRoutine = true;
        }
        else
        {
            sucessfullyConnected = true;
        }

        int receiveLoopErr = 0;
        // Receives the data from the camera
        while (!exitingRoutine && !mExiting && !getExitApp() && !(receiveLoopErr = receiveLoop())) // This will now restart when the receiveLoop exits with a 0 return code or getExitApp() returns true
        {
        }

#ifdef _WIN32
        closesocket(mTcpClientSocket);
        mTcpClientSocket = INVALID_SOCKET;
        WSACleanup();
#endif

        return 0;
    }

    /////////////////////////
    // Receive Loop reads from the camera until the camera stops recording
    // Then depending on whether or not the application is running in Service 
    // mode or not, it will either exit, or wait for the camera to start 
    // recording again, and begin saving a new R3D file.
    /////////////////////////
    int receiveLoop()
    {
#define BufferSize (24 * 1024 * 1024)
        int             bytesToRead = 0;
        int             bytesRead;
        int             rxBufIdx = 0;

        const int64_t   current_timeout_ms = 2000;

        // TCP Parameters
#ifdef _WIN32        
        std::vector<WSAPOLLFD>    tcpPollFds;
        WSAPOLLFD                   tcpPollFd;
#else
        std::vector<struct pollfd>     tcpPollFds;
        struct pollfd               tcpPollFd;
#endif
        memset(&tcpPollFd, 0, sizeof(tcpPollFd));
        tcpPollFds.push_back(tcpPollFd);

        bool            cameraIsConnected = false;
        int             sts = 0;

        R3DSDK::R3DStream2 * stream = new R3DSDK::R3DStream2(mFolder.c_str(), mReelId, mClipId);
        bytesToRead = stream->BytesToRead(mRxBuffer, rxBufIdx);

        /////////////////////////
        // Primary loop to receive data from camera
        /////////////////////////
        while (!mExiting && !getExitApp())
        {
            if (rxBufIdx + bytesToRead > BufferSize)
            {
                std::cout << "Error: reading bytesToRead (" << bytesToRead << ") will overrun the max buffer size (" << BufferSize << ").  Dumping the data and resyncing." << std::endl;
                rxBufIdx = 0;
                bytesToRead = stream->BytesToRead(mRxBuffer, rxBufIdx);
            }

            sts = 0;

            tcpPollFds[0].fd = mTcpClientSocket;
            tcpPollFds[0].events = POLLIN;
#ifdef _WIN32
            const int numTcpPollFds = WSAPoll(&tcpPollFds[0], (ULONG)tcpPollFds.size(), (INT)current_timeout_ms);
#else
            const int numTcpPollFds = poll(&tcpPollFds[0], (int)tcpPollFds.size(), (int)current_timeout_ms);
#endif

            if (getExitApp())
                break;

            if (!numTcpPollFds)
            {
                if (!cameraIsConnected)
                {
                    std::cout << "***** TCP Receive Protocol connected to the camera but the poll didn't return any sockets." << std::endl;
                }
                else
                {
                    if (!mKeepGoing)
                    {
                        std::cout << "***** Exiting TCP Receive Protocol connected to the camera but the poll didn't return any sockets.  The camera has stopped recording." << std::endl;
                        mExiting = true;
                        break;
                    }
                    if (!cameraIsConnected) // Don't spam the console if this has already been printed once
                        std::cout << "***** TCP Receive Protocol the poll didn't return any sockets." << std::endl;
                }
                if (!cameraIsConnected)
                    cameraIsConnected = true;
                continue;
            }
            else if (numTcpPollFds > 0)
            {
                if (!cameraIsConnected)
                    std::cout << "Poll revents: (" << tcpPollFds[0].revents << ") " << getPollfdEventDesc(tcpPollFds[0].revents) << ". Continuing..." << std::endl;
                if (tcpPollFds[0].revents & (POLLERR | POLLHUP))
                {
                    std::cout << "***** Exiting TCP Receive Protocol receive loop due to " << getPollfdEventDesc(tcpPollFds[0].revents) << " result from the poll request." << std::endl;
                    sts = 7;
                    break;
                }
                if (!cameraIsConnected)
                    std::cout << "***** TCP Receive Protocol connected to the camera." << std::endl;
            }
            else
            {
                int socketErr = WSAGetLastError();
                std::cout << "***** Exiting TCP Receive Protocol receive loop due to SOCKET_ERROR from the poll.  Error: " << socketErr << ". " << getErrorText(socketErr) << std::endl;
                sts = 8;
                break;
            }

            if (!cameraIsConnected)
                cameraIsConnected = true;

            // Read the TCP packet
#ifdef _WIN32            
            u_long iMode = 1;
            ioctlsocket(tcpPollFds[0].fd, FIONBIO, &iMode);
#endif
            bytesRead = recv(tcpPollFds[0].fd, mRxBuffer + rxBufIdx, bytesToRead, 0);

            rxBufIdx += bytesRead;

#if 1
            bytesToRead = stream->BytesToRead(mRxBuffer, rxBufIdx);

            if (bytesToRead == 0)
            {
                bool isDroppedFrame = false;
                const R3DSDK::CreateStatus cs = stream->WriteData(mRxBuffer, rxBufIdx, isDroppedFrame);

                if (cs > R3DSDK::CSDone)
                {
                    std::cout << "R3D write error " << cs << std::endl;
                    sts = (int)cs;
                    break;
                }
                else if (cs == R3DSDK::CSStarted)
                {
                    std::cout << "The camera started recording" << std::endl;
                }
                else if (cs == R3DSDK::CSDone)
                {
                    std::cout << "The camera stopped recording" << std::endl;
                    if (!mKeepAlive)    // If not in service mode then exit
                    {                   // deleteSteam will be called on the way out
                        sts = 9;
                        break;
                    }
                    delete stream;

                    // get ready to start the next clip
                    UpdateClipId();
                    stream = new R3DSDK::R3DStream2(mFolder.c_str(), mReelId, mClipId);
                }
                else if (cs == R3DSDK::CSFrameAdded)
                {
                    // success
                }

                if (isDroppedFrame)
                {
                    std::cout << "Warning: dropped a frame" << std::endl;
                }

                // reset for next frame
                bytesToRead = stream->BytesToRead(mRxBuffer, rxBufIdx);
            }
#else
            const bool cameraWasRecording = mCameraIsRecording;

            mKeepGoing = ProcessBuffer(mRxBuffer, BufferSize, rxBufIdx, bytesToRead, &mHandle, mFolder, mReelId, mClipId, true, mCameraIsRecording, mKeepAlive);

            if (cameraWasRecording != mCameraIsRecording)
                std::cout << "The camera " << (mCameraIsRecording ? "started" : "stopped") << " recording." << std::endl;
#endif
        }

        delete stream;

        return sts;
    }

    /////////////////////////
    // Print Usage
    /////////////////////////
    void usage(char* app)
    {
        std::cout << app << " -[c|f|i|r|s]" << std::endl;
        std::cout << "    -c <clip Id>      Starting clip ID (1 - 999)." << std::endl;
        std::cout << "    -f <path>         Folder to write the output R3D clips to." << std::endl;
        std::cout << "    -i <camera IP>    Camera IP address." << std::endl;
        std::cout << "    -r <reel Id>      Starting reel ID (1 - 999)." << std::endl;
        std::cout << "    -R <path>         Override folder to the REDR3D dyanimc library (defauls to current folder)." << std::endl;
        std::cout << "    -s                Service mode. Don't exit when camera finishes recording, continue to save additional clips." << std::endl;
    }

    /////////////////////////
    // Parses the Command Line Options
    /////////////////////////
    bool ParseCommandLine(int argc, char* argv[])
    {
        bool gotIp = false;

        for (int arg = 1; arg < argc; arg++)
        {
            if (argv[arg][0] != '-')
            {
                usage(argv[0]);
                return false;
            }
            else
            {
                if (argv[arg][1] == 'c')
                {
                    sscanf_s(argv[++arg], "%d", &mClipId);
                }
                else if (argv[arg][1] == 'r')
                {
                    sscanf_s(argv[++arg], "%d", &mReelId);
                }
                else if (argv[arg][1] == 's')
                {
                    mKeepAlive = true;
                }
                else if (argv[arg][1] == 'i')
                {
                    mHost = argv[++arg];
                    gotIp = true;
                }
                else if (argv[arg][1] == 'f')
                {
                    mFolder = argv[++arg];
                }
                else if (argv[arg][1] == 'R')
                {
                    mR3DSDKFolder = argv[++arg];
                }
                else
                {
                    usage(argv[0]);
                    return false;
                }
            }
        }
        if (!gotIp)
        {
            usage(argv[0]);
            return false;
        }
        return true;
    }

private:
    bool mExiting;
    bool mKeepGoing;

    std::string mHost;
    std::string mPort;
    std::string mFolder;
    std::string mR3DSDKFolder;
    int mClipId;
    int mReelId;
    bool mKeepAlive;

    void *mHandle;
    bool mCameraIsRecording;

    // TCP Parameters
    SOCKET mTcpClientSocket;
    bool mTcpKeepAliveEnabled;

    char mRxBuffer[BufferSize];

    void UpdateClipId()
    {
        mClipId++;

        if (mClipId > 999)
        {
            mClipId = 1;
            mReelId++;

            if (mReelId > 999)
                mReelId = 1;
        }
    }
};


int main(int argc, char* argv[])
{
    // The ReceiveStream class contains a 24MB input buffer (mRxBuffer), 
    // best to instantiate it via a new, instead of attempting to put it on the stack
    ReceiveStream *receiveStream = new ReceiveStream();

    if (!receiveStream->ParseCommandLine(argc, argv))
        return -1;

#ifdef _WIN32 
    if (!SetConsoleCtrlHandler(consoleHandler, TRUE))
        std::cout << "ERROR: Could not set control-C handler" << std::endl;
    else
        std::cout << "Press Control-C to exit." << std::endl;
#endif

    return receiveStream->Start();
}

