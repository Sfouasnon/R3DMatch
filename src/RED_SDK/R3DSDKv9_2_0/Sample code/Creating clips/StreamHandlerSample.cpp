/* R3D SDK sample code.

   This sample code and everything else included with the R3D
   SDK is Copyright (c) 2008-2026 RED Digital Cinema. All rights
   reserved. Redistribution of this sample code is prohibited!
*/

/* Sample to demonstrate saving stream coming in over tether

   Requires Boost */

#include <stdint.h>
#include <iostream>

#include <R3DSDK.h>

#include "rcp_api/rcp_api.h"
#include <R3DSDKStream.h>

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/shared_array.hpp>
#include <boost/date_time.hpp>
#include <deque>

using boost::asio::ip::udp;
using boost::asio::ip::tcp;
using namespace R3DSDK;


namespace
{
    boost::thread writeThread;

    boost::recursive_mutex rcpConnectionMutex;
    boost::recursive_mutex rcpDiscoveryMutex;
}


void* rcp_malloc(size_t bytes)
{
    return malloc(bytes);
}

void rcp_free(void *ptr)
{
    free(ptr);
}

int rcp_rand()
{
    static int seed = 1;
    if(seed)
    {
        srand(time(NULL));
        seed = 0;
    }

    return rand();
}

uint32_t rcp_timestamp()
{
    static bool shouldCalcStart = true;
    static boost::posix_time::ptime start;
    if(shouldCalcStart)
    {
        start = boost::posix_time::microsec_clock::local_time();
        shouldCalcStart = false;
    }

    boost::posix_time::ptime now = boost::posix_time::microsec_clock::local_time();
    boost::posix_time::time_duration diff = now - start;
    return diff.total_milliseconds();
}

void rcp_log(rcp_log_t severity, const rcp_camera_connection_t * con, const char * msg)
{
    (void)severity;
    (void)con;
    printf("%s", msg);
}

void rcp_mutex_lock(rcp_mutex_t id)
{
    switch(id)
    {
        case RCP_MUTEX_CONNECTION:
            rcpConnectionMutex.lock();
            break;
        case RCP_MUTEX_DISCOVERY:
            rcpDiscoveryMutex.lock();
            break;
        default:
            break;
    }
}

void rcp_mutex_unlock(rcp_mutex_t id)
{
    switch(id)
    {
        case RCP_MUTEX_CONNECTION:
            rcpConnectionMutex.unlock();
            break;
        case RCP_MUTEX_DISCOVERY:
            rcpDiscoveryMutex.unlock();
            break;
        default:
            break;
    }
}

struct Packet
{
    boost::shared_array<unsigned char> data;
    uint32_t length;
};

class Server
{
public:
    Server(rcp_camera_connection_t *con, int listenPort, const char * path);
    ~Server();

    void setReelID(unsigned int reelId);
    void setClipID(unsigned int clipId);

    void handleReceive(const boost::system::error_code &error, size_t bytes);
    void sendAck(const unsigned char *data, uint32_t len);
    void serverThread();
    void writeThread();

    void replayServer();

private:

    unsigned char m_receiveBuf[1036];

    rcp_camera_connection_t *m_connection;
    CameraStream *m_handler;

    boost::asio::io_service m_service;
    udp::socket m_socket;
    udp::endpoint m_cameraEndpoint;

    std::deque<Packet> m_writeQueue;

    boost::mutex queueMutex;

    const char * m_path;
    unsigned int m_reelId;
    unsigned int m_clipId;
};

bool traceRecord = false;
bool traceReplay = false;
FILE * trace = NULL;

void AckCallbackFunction(void *userData, const unsigned char *data, uint32_t len)
{
    if (!traceReplay)
    {
        Server *s = (Server*)userData;
        s->sendAck(data, len);
    }
}

Server::Server(rcp_camera_connection_t *con, int listenPort, const char * path)
: m_connection(con)
, m_handler(NULL)
, m_service()
, m_socket(m_service, udp::endpoint(udp::v4(), listenPort))
, m_writeQueue()
, m_path(path)
, m_reelId(1)
, m_clipId(1)
{
    m_handler = new CameraStream(&AckCallbackFunction, this);

    try
    {
        boost::asio::socket_base::receive_buffer_size recvSize;
        m_socket.get_option(recvSize);

        if(recvSize.value() < 1048576)
        {
            recvSize = boost::asio::socket_base::receive_buffer_size(1048576);
            m_socket.set_option(recvSize);
        }
    }
    catch (...)
    {
        // couldn't set receive buffer size
        printf("Error setting receive buffer size\n");
    }

    m_socket.async_receive_from(boost::asio::buffer(m_receiveBuf, 1036),
                                m_cameraEndpoint,
                                boost::bind(&Server::handleReceive,
                                            this,
                                            boost::asio::placeholders::error,
                                            boost::asio::placeholders::bytes_transferred));
}

Server::~Server()
{
    m_service.stop();
    delete m_handler;
    m_handler = NULL;
}

void Server::setReelID(unsigned int reelId)
{
    m_reelId = reelId;
}

void Server::setClipID(unsigned int clipId)
{
    m_clipId = clipId;
}

void Server::handleReceive(const boost::system::error_code &error, size_t bytes)
{
    if(!error)
    {
        VerifyStatus vs = m_handler->ProcessRdpPacket(m_receiveBuf, bytes);

        if(vs == VS_OK)
        {
            Packet p;
            p.data.reset(new unsigned char[bytes]);
            p.length = bytes;
            memcpy(p.data.get(), m_receiveBuf, bytes);
            boost::unique_lock<boost::mutex> lock(queueMutex);
            m_writeQueue.push_back(p);
        }
    }

    m_socket.async_receive_from(boost::asio::buffer(m_receiveBuf, 1036),
                                m_cameraEndpoint,
                                boost::bind(&Server::handleReceive,
                                            this,
                                            boost::asio::placeholders::error,
                                            boost::asio::placeholders::bytes_transferred));
}

void Server::sendAck(const unsigned char *data, uint32_t len)
{
    m_socket.send_to(boost::asio::buffer(data, len),
                     m_cameraEndpoint);
}


void Server::serverThread()
{
    printf("Starting server thread.\n");
    m_service.run();
    printf("Server Thread Ended.\n");
}

enum TraceType
{
    TraceStartRecord = 1,	// followed by 2 unsigned ints: reel + clip ID
    TraceDataPacket,		// followed by unsigned int size + payload
    TraceStopRecord			// followed by nothing
};

static bool stopRecord = false;

void Server::writeThread()
{
    m_writeQueue.clear();
    R3DStream stream(m_path, m_reelId, m_clipId);

    if (traceRecord)
    {
        unsigned int buffer[3];
        buffer[0] = TraceStartRecord;
        buffer[1] = m_reelId;
        buffer[2] = m_clipId;

        if (fwrite(buffer, sizeof(buffer), 1, trace) != 1)
        {
            printf("Error writing %u bytes to trace file\n", (unsigned int)sizeof(buffer));
            exit(-1);
        }
        fflush(trace);
    }

    while(!stopRecord)
    {
        while(!m_writeQueue.empty())
        {
            Packet p;
            {
                boost::unique_lock<boost::mutex> lock(queueMutex);
                p = m_writeQueue.front();
                m_writeQueue.pop_front();
            }

            if (traceRecord)
            {
                unsigned int buffer[2];
                buffer[0] = TraceDataPacket;
                buffer[1] = p.length;

                if (fwrite(buffer, sizeof(buffer), 1, trace) != 1)
                {
                    printf("Error writing %u bytes to trace file\n", (unsigned int)sizeof(buffer));
                    exit(-1);
                }

                if (fwrite(p.data.get(), p.length, 1, trace) != 1)
                {
                    printf("Error writing %u bytes to trace file\n", p.length);
                    exit(-1);
                }

                fflush(trace);
            }

            CreateStatus ws = stream.WritePacketData(p.data.get(), p.length);

            // anything higher is an error code, so bail
            if (ws > CSDone)
            {
                printf("Error writing data to clip: %d\n", ws);
                exit(-1);
                // handle error;
            }
        }

        boost::this_thread::sleep(boost::posix_time::millisec(1));
    }

    printf("Exiting write thread\n");

    if(traceRecord)
    {
        // Note: stream goes out of scope here and closes the file.
        const unsigned int closeFlag = TraceStopRecord;

        if (fwrite(&closeFlag, sizeof(closeFlag), 1, trace) != 1)
        {
            printf("Error writing 4 bytes to trace file\n");
            exit(-1);
        }
    }
}


void Server::replayServer()
{
    unsigned int traceType = 0;

    while (fread(&traceType, 4, 1, trace) == 1)
    {
        if (traceType == TraceStartRecord)
        {
            unsigned int buffer[2] = { 0 };

            if (fread(buffer, sizeof(buffer), 1, trace) != 1)
            {
                printf("Error: failed to read %u bytes from trace file\n", (unsigned int)sizeof(buffer));
                return;
            }

            setReelID(buffer[0]);
            setClipID(buffer[1]);

            if(!::writeThread.joinable())
            {
                ::writeThread = boost::thread(boost::bind(&Server::writeThread, this));
            }
        }
        else if (traceType == TraceDataPacket)
        {
            unsigned int packetSize = 0;

            if (fread(&packetSize, sizeof(packetSize), 1, trace) != 1)
            {
                printf("Error: failed to read 4 bytes from trace file\n");
                return;
            }

            if (packetSize > 0)
            {
                Packet p;
                p.data.reset(new unsigned char[packetSize]);
                p.length = packetSize;

                if (fread(p.data.get(), packetSize, 1, trace) != 1)
                {
                    printf("Error: failed to read %u bytes from trace file\n", packetSize);
                    return;
                }

                boost::unique_lock<boost::mutex> lock(queueMutex);
                m_writeQueue.push_back(p);
            }
        }
        else if (traceType == TraceStopRecord)
        {
            stopRecord = true;

            if(::writeThread.joinable())
            {
                ::writeThread.join();
            }

            stopRecord = false;
        }
        else
        {
            printf("Invalid trace type received: %u\n", traceType);
            return;
        }
    }
}

class Camera;

rcp_error_t sendDataToCameraCallback(const char *data, size_t len, void *user_data);
void cameraCurrentStateData(const rcp_state_data_t* data, void *user_data);

class Camera
{
public:
    Camera();
    ~Camera();

    void connect(const std::string &ip);
    bool isConnected() const { return m_connectionState == RCP_CONNECTION_STATE_CONNECTED; }
    void setConnectionState(rcp_connection_state_t state);
    void setCamInfo(const rcp_cam_info_t *camInfo);

    void setComputerIP(const std::string &ip);

    void sendCommand(const std::string &command);

    rcp_camera_connection_t* getConnection() const { return m_connection; }

private:
    void run();
    void handleConnect(const boost::system::error_code &e);
    void handleRead(const boost::system::error_code &e, size_t bytesTransferred);
    void writePacket(const std::string &command);
    void handleWrite(const boost::system::error_code &e, size_t bytesTransferred);

    boost::asio::io_service m_service;
    boost::shared_ptr<tcp::resolver> m_resolver;
    tcp::resolver::query m_query;
    tcp::resolver::iterator m_queryIter;
    boost::shared_ptr<tcp::socket> m_socket;

    boost::thread m_camThread;

    unsigned char m_data[4096];

    rcp_camera_connection_info_t m_connInfo;
    rcp_camera_connection_t *m_connection;
    rcp_connection_state_t m_connectionState;
    const rcp_cam_info_t *m_camInfo;
};

Camera::Camera()
: m_service()
, m_resolver(new tcp::resolver(m_service))
, m_query("127.0.0.1", "1111")
, m_socket()
{
    memset(&m_connInfo, 0, sizeof(rcp_camera_connection_info_t));

    m_connInfo.send_data_to_camera_cb = ::sendDataToCameraCallback;
    m_connInfo.send_data_to_camera_cb_user_data = this;

    m_connInfo.state_cb = ::cameraCurrentStateData;
    m_connInfo.state_cb_user_data = this;
}

Camera::~Camera()
{

    m_service.stop();

    if(m_socket)
    {
        m_socket->close();
    }

    m_camThread.join();
}

void Camera::connect(const std::string &ip)
{
    m_service.reset();
    m_socket.reset(new tcp::socket(m_service));
    m_resolver.reset(new tcp::resolver(m_service));
    m_query = tcp::resolver::query(ip, "1111");
    m_queryIter = m_resolver->resolve(m_query);

    boost::asio::async_connect(*m_socket, m_queryIter,
                               boost::bind(&Camera::handleConnect, this,
                                           boost::asio::placeholders::error));


    m_camThread = boost::thread(boost::bind(&Camera::run, this));
}

void Camera::setConnectionState(rcp_connection_state_t state)
{
    m_connectionState = state;
}

void Camera::setCamInfo(const rcp_cam_info_t *camInfo)
{
    m_camInfo = camInfo;
}

void Camera::setComputerIP(const std::string &ip)
{
    if(m_connectionState != RCP_CONNECTION_STATE_CONNECTED)
        return;

    rcp_set_str(m_connection, RCP_PARAM_TETHERED_SERVER_ADDRESS, ip.c_str());
}

void Camera::run()
{
    printf("Camera Thread Started\n");
    m_service.run();
    printf("Camera Thread Ended\n");
}

void Camera::handleConnect(const boost::system::error_code &e)
{
    if(!e)
    {
        m_socket->async_read_some(boost::asio::buffer(m_data, 4096),
                                  boost::bind(&Camera::handleRead, this,
                                              boost::asio::placeholders::error,
                                              boost::asio::placeholders::bytes_transferred));

        m_connection = rcp_create_camera_connection(&m_connInfo);
    }
    else
    {
        printf("Error connecting to camera: %s", e.message().c_str());
    }
}

void Camera::handleRead(const boost::system::error_code &e, size_t bytesTransferred)
{
    if(!e)
    {
        std::string message((char*)m_data, 0, bytesTransferred);
        rcp_process_data(m_connection, message.c_str(), message.size());

        m_socket->async_read_some(boost::asio::buffer(m_data, 4096),
                                  boost::bind(&Camera::handleRead, this,
                                              boost::asio::placeholders::error,
                                              boost::asio::placeholders::bytes_transferred));
    }
    else
    {
        printf("Error reading message from camera: %s", e.message().c_str());
    }
}

void Camera::sendCommand(const std::string &cmd)
{
    m_service.post(boost::bind(&Camera::writePacket, this, cmd));
}

void Camera::writePacket(const std::string &cmd)
{
    boost::asio::async_write(*m_socket, boost::asio::buffer(cmd.c_str(), cmd.size()),
                             boost::bind(&Camera::handleWrite, this,
                                         boost::asio::placeholders::error,
                                         boost::asio::placeholders::bytes_transferred));
}

void Camera::handleWrite(const boost::system::error_code &e, size_t bytesTransferred)
{
    if(e)
    {
        printf("Error writing data to camera: %s", e.message().c_str());
    }
}

rcp_error_t sendDataToCameraCallback(const char *data, size_t len, void *user_data)
{
    Camera *camera = (Camera*)user_data;
    std::string command(data, len);
    camera->sendCommand(command);
    return RCP_SUCCESS;
}

void cameraCurrentStateData(const rcp_state_data_t* data, void *user_data)
{
    Camera *camera = (Camera*)user_data;
    camera->setConnectionState(data->state);

    switch(data->state)
    {
        case RCP_CONNECTION_STATE_INIT:
            break;
        case RCP_CONNECTION_STATE_CONNECTED:
            camera->setCamInfo(data->cam_info);
            break;
        case RCP_CONNECTION_STATE_ERROR_RCP_VERSION_MISMATCH:
            printf("rcp version mismatch\n");
            break;
        case RCP_CONNECTION_STATE_ERROR_RCP_PARAMETER_SET_VERSION_MISMATCH:
            printf("parameter set version mismatch\n");
            break;
        case RCP_CONNECTION_STATE_COMMUNICATION_ERROR:
            printf("communication error\n");
            break;
    }
}

int run_server(int argc, char**argv)
{
    if ((argc < 6) || (argc > 7))
    {
        printf("Usage: %s Output_folder Reel_ID Clip_ID Camera_IP Computer_IP [tracelog]\n\n", argv[0]);
        printf("Output_folder: location to store the output\n");
        printf("Reel_ID      : reel ID sent to camera, must be in range 1 - 999\n");
        printf("Clip_ID      : clip ID sent to camera, must be in range 1 - 999\n");
        printf("Camera_IP    : ip address of the camera.\n");
        printf("Computer_IP  : ip address of the computer.\n");
        printf("tracelog     : capture/replay camera stream for debugging\n");
        printf("               - records log if file doesn't exist\n");
        printf("               - plays back log if file exists (no camera needed)\n");
        return -1;
    }


    if (argc == 7)
    {
        trace = fopen(argv[6], "rb");

        if (trace != NULL)
        {
            traceReplay = true;
            printf("Mode: trace replay\n");
        }
        else
        {
            trace = fopen(argv[6], "wb");

            if (trace == NULL)
            {
                printf("Error: unable to create or replay from %s\n", argv[1]);
                return -1;
            }

            traceRecord = true;
            printf("Mode: trace record\n");
        }
    }
    else
        printf("Mode: normal\n");

    Camera c;
    printf("Connecting to camera at: %s\n", argv[4]);
    c.connect(argv[4]);

    int retryCount = 0;
    while(true)
    {
        if(retryCount > 5)
        {
            printf("Error connecting to camera...\n");
            return -2;
        }

        if(c.isConnected())
            break;

        boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
        ++retryCount;
    }

    printf("Connected.\n");

    unsigned int reelId = atoi(argv[2]);
    unsigned int clipId = atoi(argv[3]);

    rcp_set_int(c.getConnection(), RCP_PARAM_TETHERED_SERVER_REEL_NO, reelId);
    rcp_set_int(c.getConnection(), RCP_PARAM_TETHERED_SERVER_CLIP_NO, clipId);
    rcp_set_int(c.getConnection(), RCP_PARAM_PRIMARY_STORAGE, PRIMARY_STORAGE_NETWORK);

    CameraUIState camState(rcp_set_int, rcp_set_uint, c.getConnection());
    camState.SetReelId(atoi(argv[2]));
    camState.SetClipId(atoi(argv[3]));

    c.setComputerIP(argv[5]);

    Server s(c.getConnection(), 1113, argv[1]);
    s.setReelID(reelId);
    s.setClipID(clipId);

    if (!traceReplay)
    {
        boost::thread serverThread(boost::bind(&Server::serverThread, &s));
    }

    if (traceReplay)
    {
        s.replayServer();
        boost::this_thread::sleep(boost::posix_time::millisec(1000));
        return 0;
    }
    else
    {
        bool recording = false;
        while(true)
        {
            printf("\nr) Record\ns) Stop Record\nq) Quit\nOption: ");
            std::string option;
            std::getline(std::cin, option);
            switch(option[0])
            {
                case 'q':
                    printf("Quitting.\n");
                    stopRecord = true;
                    writeThread.join();

                    return 0;
                case 'r':
                {
                    printf("Starting Record.\n");
                    if(recording)
                    {
                        printf("Record already in progress.\n");
                        break;
                    }

                    stopRecord = false;

                    if(!writeThread.joinable())
                    {
                        writeThread = boost::thread(boost::bind(&Server::writeThread, &s));
                    }

                    rcp_set_int(c.getConnection(), RCP_PARAM_RECORD_STATE, SET_RECORD_STATE_START);
                    camState.SetUIRecordState(true);
                    recording = true;
                    break;
                }
                case 's':
                {
                    printf("Stopping Record.\n");
                    if(!recording)
                        break;

                    rcp_set_int(c.getConnection(), RCP_PARAM_RECORD_STATE, SET_RECORD_STATE_STOP);
                    stopRecord = true;
                    recording = false;

                    if(writeThread.joinable())
                    {
                        writeThread.join();
                    }
                    camState.SetUIRecordState(false);
                    clipId += 1;
                    rcp_set_int(c.getConnection(), RCP_PARAM_TETHERED_SERVER_CLIP_NO, clipId);
                    s.setClipID(clipId);
                    break;
                }
                default:
                {
                    printf("\nUnknown Option\n");
                }
            }
        }
    }
    return 0;
}

int main(int argc, char **argv)
{
    if (R3DSDK::InitializeSdk(".", 0) != ISInitializeOK)
	{
		printf("Error initializing SDK. Dynamic library not found?\n");
		return -1;
	}

    int retval = run_server(argc, argv);

    R3DSDK::FinalizeSdk();
    return retval;
}
