#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "rcp_api.h"

#define R3DMATCH_CONN_TIMEOUT_MS 4000
#define R3DMATCH_IO_POLL_MS 100
#define R3DMATCH_QUEUE_CAPACITY 65536
#define R3DMATCH_MANUAL_RESPONSE_CAPACITY 256
#define R3DMATCH_PREAMBLE_WAIT_MS 2000

static pthread_mutex_t g_rcp_mutexes[RCP_MUTEX_COUNT];
static pthread_once_t g_rcp_mutex_once = PTHREAD_ONCE_INIT;

static void r3dmatch_init_rcp_mutexes(void)
{
    int i;
    for (i = 0; i < RCP_MUTEX_COUNT; ++i)
    {
        pthread_mutex_init(&g_rcp_mutexes[i], NULL);
    }
}

void *rcp_malloc(size_t nbytes)
{
    return malloc(nbytes);
}

void rcp_free(void *memory)
{
    free(memory);
}

void rcp_mutex_lock(rcp_mutex_t id)
{
    pthread_once(&g_rcp_mutex_once, r3dmatch_init_rcp_mutexes);
    if ((int)id < 0 || id >= RCP_MUTEX_COUNT)
    {
        return;
    }
    pthread_mutex_lock(&g_rcp_mutexes[(int)id]);
}

void rcp_mutex_unlock(rcp_mutex_t id)
{
    pthread_once(&g_rcp_mutex_once, r3dmatch_init_rcp_mutexes);
    if ((int)id < 0 || id >= RCP_MUTEX_COUNT)
    {
        return;
    }
    pthread_mutex_unlock(&g_rcp_mutexes[(int)id]);
}

void rcp_log(rcp_log_t severity, const rcp_camera_connection_t *con, const char *msg)
{
    const char *label = "INFO";
    (void)con;
    switch (severity)
    {
        case RCP_LOG_WARNING:
            label = "WARN";
            break;
        case RCP_LOG_ERROR:
            label = "ERROR";
            break;
        default:
            label = "INFO";
            break;
    }
    if (!msg)
    {
        msg = "";
    }
    fprintf(stderr, "[RCP2 %s] %s\n", label, msg);
    fflush(stderr);
}

int rcp_rand(void)
{
    return rand();
}

uint32_t rcp_timestamp(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint32_t)(((uint64_t)tv.tv_sec * 1000ULL) + ((uint64_t)tv.tv_usec / 1000ULL));
}

typedef struct
{
    int socket_fd;
    int stop_requested;
    int transport_error;
    int connect_complete;

    char last_error[512];

    pthread_mutex_t mutex;
    pthread_cond_t cond;
    pthread_t rx_thread;
    pthread_t create_thread;

    rcp_camera_connection_t *con;
    rcp_connection_state_t connection_state;
    int connection_state_valid;
    int32_t parameter_set_version_major;
    int32_t parameter_set_version_minor;
    int parameter_set_version_valid;
    int parameter_set_newer;

    char camera_id[64];
    char camera_pin[64];
    char camera_type[128];
    char camera_version[128];

    struct
    {
        int has_value;
        int32_t value;
        uint64_t seq;
    } exposure_adjust;
    struct
    {
        int has_value;
        int32_t value;
        uint64_t seq;
    } color_temperature;
    struct
    {
        int has_value;
        int32_t value;
        uint64_t seq;
    } tint;

    unsigned char pending_bytes[R3DMATCH_QUEUE_CAPACITY];
    size_t pending_len;
    char last_inbound_preview[128];
    int raw_logging_enabled;
    char manual_response[R3DMATCH_MANUAL_RESPONSE_CAPACITY];
    int resetid_seen;
    int who_seen;
} r3dmatch_rcp_ctx_t;

void r3dmatch_rcp_close(void *handle);

static void r3dmatch_set_error(r3dmatch_rcp_ctx_t *ctx, const char *message)
{
    if (!ctx || !message)
    {
        return;
    }
    pthread_mutex_lock(&ctx->mutex);
    snprintf(ctx->last_error, sizeof(ctx->last_error), "%s", message);
    pthread_cond_broadcast(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);
}

static void r3dmatch_set_errno_error(r3dmatch_rcp_ctx_t *ctx, const char *prefix)
{
    char buf[512];
    snprintf(buf, sizeof(buf), "%s: %s", prefix, strerror(errno));
    r3dmatch_set_error(ctx, buf);
}

static void r3dmatch_capture_inbound_preview(r3dmatch_rcp_ctx_t *ctx, const unsigned char *buffer, size_t len)
{
    if (!ctx || !buffer || len == 0)
    {
        return;
    }

    size_t out_index = 0;
    size_t limit = len < 96 ? len : 96;
    size_t i;
    for (i = 0; i < limit && out_index + 1 < sizeof(ctx->last_inbound_preview); ++i)
    {
        unsigned char ch = buffer[i];
        if (ch == '\r' || ch == '\n' || ch == '\t')
        {
            if (out_index > 0 && ctx->last_inbound_preview[out_index - 1] != ' ')
            {
                ctx->last_inbound_preview[out_index++] = ' ';
            }
            continue;
        }
        if (ch >= 32 && ch <= 126)
        {
            ctx->last_inbound_preview[out_index++] = (char)ch;
        }
        else
        {
            ctx->last_inbound_preview[out_index++] = '.';
        }
    }
    ctx->last_inbound_preview[out_index] = '\0';
}

static int r3dmatch_truthy_env(const char *name)
{
    const char *value = getenv(name);
    if (!value || value[0] == '\0')
    {
        return 0;
    }
    return strcmp(value, "0") != 0;
}

static void r3dmatch_copy_env_response(char *dest, size_t dest_size, const char *name)
{
    const char *value = getenv(name);
    size_t src_len;
    size_t out_index = 0;
    size_t i;

    if (!dest || dest_size == 0)
    {
        return;
    }
    dest[0] = '\0';
    if (!value || value[0] == '\0')
    {
        return;
    }

    src_len = strlen(value);
    for (i = 0; i < src_len && out_index + 1 < dest_size; ++i)
    {
        char ch = value[i];
        if (ch == '\\' && (i + 1) < src_len)
        {
            char next = value[i + 1];
            if (next == 'n')
            {
                dest[out_index++] = '\n';
                i += 1;
                continue;
            }
            if (next == 'r')
            {
                dest[out_index++] = '\r';
                i += 1;
                continue;
            }
            if (next == 't')
            {
                dest[out_index++] = '\t';
                i += 1;
                continue;
            }
        }
        dest[out_index++] = ch;
    }
    if (out_index > 0 && dest[out_index - 1] != '\n' && out_index + 1 < dest_size)
    {
        dest[out_index++] = '\n';
    }
    dest[out_index] = '\0';
}

static void r3dmatch_format_wire_preview(const unsigned char *buffer, size_t len, char *text_buf, size_t text_buf_size, char *hex_buf, size_t hex_buf_size)
{
    size_t i;
    size_t text_index = 0;
    size_t hex_index = 0;
    size_t limit = len < 96 ? len : 96;

    if (text_buf && text_buf_size > 0)
    {
        text_buf[0] = '\0';
    }
    if (hex_buf && hex_buf_size > 0)
    {
        hex_buf[0] = '\0';
    }

    for (i = 0; i < limit; ++i)
    {
        unsigned char ch = buffer[i];
        if (text_buf && text_index + 1 < text_buf_size)
        {
            if (ch == '\r' || ch == '\n' || ch == '\t')
            {
                if (text_index > 0 && text_buf[text_index - 1] != ' ')
                {
                    text_buf[text_index++] = ' ';
                }
            }
            else if (ch >= 32 && ch <= 126)
            {
                text_buf[text_index++] = (char)ch;
            }
            else
            {
                text_buf[text_index++] = '.';
            }
        }
        if (hex_buf && hex_index + 2 < hex_buf_size)
        {
            int written = snprintf(hex_buf + hex_index, hex_buf_size - hex_index, "%02x", (unsigned int)ch);
            if (written < 0)
            {
                break;
            }
            hex_index += (size_t)written;
        }
    }

    if (text_buf && text_buf_size > 0)
    {
        text_buf[text_index < text_buf_size ? text_index : text_buf_size - 1] = '\0';
    }
    if (hex_buf && hex_buf_size > 0)
    {
        hex_buf[hex_index < hex_buf_size ? hex_index : hex_buf_size - 1] = '\0';
    }
}

static void r3dmatch_log_wire(r3dmatch_rcp_ctx_t *ctx, const char *direction, const unsigned char *buffer, size_t len)
{
    char text_buf[128];
    char hex_buf[256];
    if (!ctx || !ctx->raw_logging_enabled || !buffer || len == 0)
    {
        return;
    }
    r3dmatch_format_wire_preview(buffer, len, text_buf, sizeof(text_buf), hex_buf, sizeof(hex_buf));
    fprintf(stderr, "[RCP2 WIRE] %s len=%zu text=\"%s\" hex=%s\n", direction ? direction : "?", len, text_buf, hex_buf);
    fflush(stderr);
}

static int r3dmatch_send_socket_data(r3dmatch_rcp_ctx_t *ctx, const char *data, size_t len, const char *error_prefix)
{
    size_t sent = 0;
    while (sent < len)
    {
        ssize_t rc = send(ctx->socket_fd, data + sent, len - sent, 0);
        if (rc < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            r3dmatch_set_errno_error(ctx, error_prefix ? error_prefix : "send failed");
            return 0;
        }
        sent += (size_t)rc;
    }
    return 1;
}

static int r3dmatch_is_ascii_protocol_line(const unsigned char *buffer, size_t len)
{
    return buffer && len >= 6 && memcmp(buffer, "#$CAM:", 6) == 0;
}

static int r3dmatch_handle_ascii_protocol_line(r3dmatch_rcp_ctx_t *ctx, const unsigned char *buffer, size_t len)
{
    const char resetid_line[] = "#$CAM:S:RESETID:\n";
    const char who_line[] = "#$CAM:G:WHO:\n";

    if (!ctx || !buffer || len == 0 || !r3dmatch_is_ascii_protocol_line(buffer, len))
    {
        return 0;
    }

    r3dmatch_log_wire(ctx, "IN", buffer, len);

    if (len == sizeof(resetid_line) - 1 && memcmp(buffer, resetid_line, sizeof(resetid_line) - 1) == 0)
    {
        ctx->resetid_seen = 1;
        fprintf(stderr, "[RCP2 WIRE] swallowing pre-SDK RESETID preamble\n");
        fflush(stderr);
        return 1;
    }

    if (ctx->manual_response[0] != '\0' &&
        len == sizeof(who_line) - 1 &&
        memcmp(buffer, who_line, sizeof(who_line) - 1) == 0)
    {
        ctx->who_seen = 1;
        r3dmatch_log_wire(ctx, "OUT-MANUAL", (const unsigned char *)ctx->manual_response, strlen(ctx->manual_response));
        if (!r3dmatch_send_socket_data(ctx, ctx->manual_response, strlen(ctx->manual_response), "manual handshake send failed"))
        {
            return 1;
        }
    }
    else if (len == sizeof(who_line) - 1 && memcmp(buffer, who_line, sizeof(who_line) - 1) == 0)
    {
        ctx->who_seen = 1;
        fprintf(stderr, "[RCP2 WIRE] observed WHO preamble; delaying SDK traffic until preamble completes\n");
        fflush(stderr);
        return 1;
    }

    return 0;
}

static int r3dmatch_connect_socket(const char *host, int port, int timeout_ms, char *error_buf, size_t error_buf_size)
{
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
    {
        snprintf(error_buf, error_buf_size, "socket failed: %s", strerror(errno));
        return -1;
    }

    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0)
    {
        snprintf(error_buf, error_buf_size, "fcntl failed: %s", strerror(errno));
        close(fd);
        return -1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, host, &addr.sin_addr) != 1)
    {
        snprintf(error_buf, error_buf_size, "invalid IP address: %s", host);
        close(fd);
        return -1;
    }

    int rc = connect(fd, (struct sockaddr *)&addr, sizeof(addr));
    if (rc < 0 && errno != EINPROGRESS)
    {
        snprintf(error_buf, error_buf_size, "connect failed: %s", strerror(errno));
        close(fd);
        return -1;
    }

    fd_set wfds;
    FD_ZERO(&wfds);
    FD_SET(fd, &wfds);
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    rc = select(fd + 1, NULL, &wfds, NULL, &tv);
    if (rc <= 0)
    {
        snprintf(error_buf, error_buf_size, "connect timeout to %s:%d", host, port);
        close(fd);
        return -1;
    }

    int so_error = 0;
    socklen_t so_error_len = sizeof(so_error);
    if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_error, &so_error_len) < 0 || so_error != 0)
    {
        snprintf(error_buf, error_buf_size, "connect failed: %s", strerror(so_error ? so_error : errno));
        close(fd);
        return -1;
    }

    if (fcntl(fd, F_SETFL, flags) < 0)
    {
        snprintf(error_buf, error_buf_size, "fcntl restore failed: %s", strerror(errno));
        close(fd);
        return -1;
    }

    return fd;
}

static void r3dmatch_store_int_value(r3dmatch_rcp_ctx_t *ctx, rcp_param_t id, int32_t value)
{
    if (!ctx)
    {
        return;
    }
    switch (id)
    {
        case RCP_PARAM_EXPOSURE_ADJUST:
            ctx->exposure_adjust.has_value = 1;
            ctx->exposure_adjust.value = value;
            ctx->exposure_adjust.seq += 1;
            break;
        case RCP_PARAM_COLOR_TEMPERATURE:
            ctx->color_temperature.has_value = 1;
            ctx->color_temperature.value = value;
            ctx->color_temperature.seq += 1;
            break;
        case RCP_PARAM_TINT:
            ctx->tint.has_value = 1;
            ctx->tint.value = value;
            ctx->tint.seq += 1;
            break;
        default:
            break;
    }
}

static rcp_error_t r3dmatch_send_data(const char *data, size_t len, void *user_data)
{
    r3dmatch_rcp_ctx_t *ctx = (r3dmatch_rcp_ctx_t *)user_data;
    if (!ctx || ctx->socket_fd < 0)
    {
        return RCP_ERROR_SEND_DATA_TO_CAM_FAILED;
    }

    r3dmatch_log_wire(ctx, "OUT-SDK", (const unsigned char *)data, len);
    if (!r3dmatch_send_socket_data(ctx, data, len, "send failed"))
    {
        return RCP_ERROR_SEND_DATA_TO_CAM_FAILED;
    }
    return RCP_SUCCESS;
}

static void r3dmatch_int_received(const rcp_cur_int_cb_data_t *data, void *user_data)
{
    r3dmatch_rcp_ctx_t *ctx = (r3dmatch_rcp_ctx_t *)user_data;
    if (!ctx || !data || !data->cur_val_valid)
    {
        return;
    }
    pthread_mutex_lock(&ctx->mutex);
    r3dmatch_store_int_value(ctx, data->id, data->cur_val);
    pthread_cond_broadcast(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);
}

static void r3dmatch_state_updated(const rcp_state_data_t *data, void *user_data)
{
    r3dmatch_rcp_ctx_t *ctx = (r3dmatch_rcp_ctx_t *)user_data;
    if (!ctx || !data)
    {
        return;
    }
    pthread_mutex_lock(&ctx->mutex);
    ctx->connection_state = data->state;
    ctx->connection_state_valid = 1;
    ctx->parameter_set_version_major = data->parameter_set_version_major;
    ctx->parameter_set_version_minor = data->parameter_set_version_minor;
    ctx->parameter_set_version_valid = data->parameter_set_version_valid;
    ctx->parameter_set_newer = data->parameter_set_newer;
    if (data->cam_info)
    {
        snprintf(ctx->camera_id, sizeof(ctx->camera_id), "%s", data->cam_info->id);
        snprintf(ctx->camera_pin, sizeof(ctx->camera_pin), "%s", data->cam_info->pin);
        snprintf(ctx->camera_type, sizeof(ctx->camera_type), "%s", data->cam_info->type);
        snprintf(ctx->camera_version, sizeof(ctx->camera_version), "%s", data->cam_info->version);
    }
    pthread_cond_broadcast(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);
}

static void *r3dmatch_rx_loop(void *opaque)
{
    r3dmatch_rcp_ctx_t *ctx = (r3dmatch_rcp_ctx_t *)opaque;
    unsigned char buffer[4096];
    while (!ctx->stop_requested)
    {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(ctx->socket_fd, &rfds);
        struct timeval tv;
        tv.tv_sec = 0;
        tv.tv_usec = R3DMATCH_IO_POLL_MS * 1000;
        int rc = select(ctx->socket_fd + 1, &rfds, NULL, NULL, &tv);
        if (rc < 0)
        {
            if (errno == EINTR)
            {
                continue;
            }
            pthread_mutex_lock(&ctx->mutex);
            ctx->transport_error = 1;
            snprintf(ctx->last_error, sizeof(ctx->last_error), "select failed: %s", strerror(errno));
            pthread_cond_broadcast(&ctx->cond);
            pthread_mutex_unlock(&ctx->mutex);
            break;
        }
        if (rc == 0)
        {
            continue;
        }

        ssize_t received = recv(ctx->socket_fd, buffer, sizeof(buffer), 0);
        if (received == 0)
        {
            pthread_mutex_lock(&ctx->mutex);
            ctx->transport_error = 1;
            if (ctx->last_inbound_preview[0] != '\0')
            {
                snprintf(
                    ctx->last_error,
                    sizeof(ctx->last_error),
                    "camera closed the connection after sending: %s",
                    ctx->last_inbound_preview
                );
            }
            else
            {
                snprintf(ctx->last_error, sizeof(ctx->last_error), "camera closed the connection");
            }
            pthread_cond_broadcast(&ctx->cond);
            pthread_mutex_unlock(&ctx->mutex);
            break;
        }
        if (received < 0)
        {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
            {
                continue;
            }
            pthread_mutex_lock(&ctx->mutex);
            ctx->transport_error = 1;
            snprintf(ctx->last_error, sizeof(ctx->last_error), "recv failed: %s", strerror(errno));
            pthread_cond_broadcast(&ctx->cond);
            pthread_mutex_unlock(&ctx->mutex);
            break;
        }

        pthread_mutex_lock(&ctx->mutex);
        r3dmatch_capture_inbound_preview(ctx, buffer, (size_t)received);
        if (r3dmatch_handle_ascii_protocol_line(ctx, buffer, (size_t)received))
        {
            pthread_cond_broadcast(&ctx->cond);
        }
        else if (ctx->con)
        {
            rcp_process_data(ctx->con, (const char *)buffer, (size_t)received);
        }
        else if ((ctx->pending_len + (size_t)received) <= sizeof(ctx->pending_bytes))
        {
            memcpy(ctx->pending_bytes + ctx->pending_len, buffer, (size_t)received);
            ctx->pending_len += (size_t)received;
        }
        pthread_mutex_unlock(&ctx->mutex);
    }
    return NULL;
}

static void *r3dmatch_create_loop(void *opaque)
{
    r3dmatch_rcp_ctx_t *ctx = (r3dmatch_rcp_ctx_t *)opaque;
    rcp_camera_connection_info_t info;
    memset(&info, 0, sizeof(info));
    info.client_name = "R3DMatch";
    info.client_version = "1.0";
    info.client_user = "R3DMatch calibration";
    info.send_data_to_camera_cb = r3dmatch_send_data;
    info.send_data_to_camera_cb_user_data = ctx;
    info.cur_int_cb = r3dmatch_int_received;
    info.cur_int_cb_user_data = ctx;
    info.state_cb = r3dmatch_state_updated;
    info.state_cb_user_data = ctx;

    rcp_camera_connection_t *con = rcp_create_camera_connection(&info);
    pthread_mutex_lock(&ctx->mutex);
    ctx->con = con;
    ctx->connect_complete = 1;
    if (ctx->con && ctx->pending_len > 0)
    {
        rcp_process_data(ctx->con, (const char *)ctx->pending_bytes, ctx->pending_len);
        ctx->pending_len = 0;
    }
    pthread_cond_broadcast(&ctx->cond);
    pthread_mutex_unlock(&ctx->mutex);
    return NULL;
}

static int r3dmatch_wait_for_state(r3dmatch_rcp_ctx_t *ctx, rcp_connection_state_t desired_state, int timeout_ms)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L)
    {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000L;
    }

    pthread_mutex_lock(&ctx->mutex);
    while (!ctx->transport_error)
    {
        if (ctx->connection_state_valid)
        {
            if (ctx->connection_state == desired_state)
            {
                pthread_mutex_unlock(&ctx->mutex);
                return 1;
            }
            if (ctx->connection_state == RCP_CONNECTION_STATE_ERROR_RCP_VERSION_MISMATCH ||
                ctx->connection_state == RCP_CONNECTION_STATE_ERROR_RCP_PARAMETER_SET_VERSION_MISMATCH ||
                ctx->connection_state == RCP_CONNECTION_STATE_RCP_DISABLED_ON_INTERFACE ||
                ctx->connection_state == RCP_CONNECTION_STATE_COMMUNICATION_ERROR)
            {
                pthread_mutex_unlock(&ctx->mutex);
                return 0;
            }
        }
        int rc = pthread_cond_timedwait(&ctx->cond, &ctx->mutex, &ts);
        if (rc == ETIMEDOUT)
        {
            pthread_mutex_unlock(&ctx->mutex);
            return 0;
        }
    }
    pthread_mutex_unlock(&ctx->mutex);
    return 0;
}

static void r3dmatch_wait_for_preamble(r3dmatch_rcp_ctx_t *ctx, int timeout_ms)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L)
    {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000L;
    }

    pthread_mutex_lock(&ctx->mutex);
    while (!ctx->transport_error && !ctx->who_seen)
    {
        int rc = pthread_cond_timedwait(&ctx->cond, &ctx->mutex, &ts);
        if (rc == ETIMEDOUT)
        {
            break;
        }
    }
    pthread_mutex_unlock(&ctx->mutex);
}

static int r3dmatch_wait_for_param_seq(r3dmatch_rcp_ctx_t *ctx, rcp_param_t id, uint64_t baseline_seq, int timeout_ms, int32_t *out_value)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += timeout_ms / 1000;
    ts.tv_nsec += (timeout_ms % 1000) * 1000000L;
    if (ts.tv_nsec >= 1000000000L)
    {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000L;
    }

    pthread_mutex_lock(&ctx->mutex);
    while (!ctx->transport_error)
    {
        uint64_t seq = 0;
        int has_value = 0;
        int32_t value = 0;
        if (id == RCP_PARAM_EXPOSURE_ADJUST)
        {
            seq = ctx->exposure_adjust.seq;
            has_value = ctx->exposure_adjust.has_value;
            value = ctx->exposure_adjust.value;
        }
        else if (id == RCP_PARAM_COLOR_TEMPERATURE)
        {
            seq = ctx->color_temperature.seq;
            has_value = ctx->color_temperature.has_value;
            value = ctx->color_temperature.value;
        }
        else if (id == RCP_PARAM_TINT)
        {
            seq = ctx->tint.seq;
            has_value = ctx->tint.has_value;
            value = ctx->tint.value;
        }
        if (has_value && seq > baseline_seq)
        {
            if (out_value)
            {
                *out_value = value;
            }
            pthread_mutex_unlock(&ctx->mutex);
            return 1;
        }
        int rc = pthread_cond_timedwait(&ctx->cond, &ctx->mutex, &ts);
        if (rc == ETIMEDOUT)
        {
            pthread_mutex_unlock(&ctx->mutex);
            return 0;
        }
    }
    pthread_mutex_unlock(&ctx->mutex);
    return 0;
}

static uint64_t r3dmatch_current_seq(r3dmatch_rcp_ctx_t *ctx, rcp_param_t id)
{
    uint64_t seq = 0;
    pthread_mutex_lock(&ctx->mutex);
    if (id == RCP_PARAM_EXPOSURE_ADJUST)
    {
        seq = ctx->exposure_adjust.seq;
    }
    else if (id == RCP_PARAM_COLOR_TEMPERATURE)
    {
        seq = ctx->color_temperature.seq;
    }
    else if (id == RCP_PARAM_TINT)
    {
        seq = ctx->tint.seq;
    }
    pthread_mutex_unlock(&ctx->mutex);
    return seq;
}

static int r3dmatch_get_param_int(r3dmatch_rcp_ctx_t *ctx, rcp_param_t id, int timeout_ms, int32_t *out_value)
{
    if (!ctx || !ctx->con)
    {
        return 0;
    }
    uint64_t baseline = r3dmatch_current_seq(ctx, id);
    if (rcp_get(ctx->con, id) != RCP_SUCCESS)
    {
        return 0;
    }
    return r3dmatch_wait_for_param_seq(ctx, id, baseline, timeout_ms, out_value);
}

static int r3dmatch_set_param_int(r3dmatch_rcp_ctx_t *ctx, rcp_param_t id, int32_t value)
{
    if (!ctx || !ctx->con)
    {
        return 0;
    }
    return rcp_set_int(ctx->con, id, value) == RCP_SUCCESS;
}

void *r3dmatch_rcp_open(const char *host, int port, int timeout_ms, char *error_buf, size_t error_buf_size)
{
    if (!host)
    {
        snprintf(error_buf, error_buf_size, "host is required");
        return NULL;
    }

    r3dmatch_rcp_ctx_t *ctx = (r3dmatch_rcp_ctx_t *)calloc(1, sizeof(r3dmatch_rcp_ctx_t));
    if (!ctx)
    {
        snprintf(error_buf, error_buf_size, "failed to allocate context");
        return NULL;
    }
    ctx->socket_fd = -1;
    pthread_mutex_init(&ctx->mutex, NULL);
    pthread_cond_init(&ctx->cond, NULL);
    ctx->raw_logging_enabled = r3dmatch_truthy_env("R3DMATCH_RCP2_RAW_LOG");
    r3dmatch_copy_env_response(ctx->manual_response, sizeof(ctx->manual_response), "R3DMATCH_RCP2_MANUAL_RESPONSE");

    ctx->socket_fd = r3dmatch_connect_socket(host, port, timeout_ms > 0 ? timeout_ms : R3DMATCH_CONN_TIMEOUT_MS, error_buf, error_buf_size);
    if (ctx->socket_fd < 0)
    {
        pthread_mutex_destroy(&ctx->mutex);
        pthread_cond_destroy(&ctx->cond);
        free(ctx);
        return NULL;
    }

    if (pthread_create(&ctx->rx_thread, NULL, r3dmatch_rx_loop, ctx) != 0)
    {
        snprintf(error_buf, error_buf_size, "failed to start receive thread");
        close(ctx->socket_fd);
        pthread_mutex_destroy(&ctx->mutex);
        pthread_cond_destroy(&ctx->cond);
        free(ctx);
        return NULL;
    }

    r3dmatch_wait_for_preamble(ctx, R3DMATCH_PREAMBLE_WAIT_MS);

    if (pthread_create(&ctx->create_thread, NULL, r3dmatch_create_loop, ctx) != 0)
    {
        snprintf(error_buf, error_buf_size, "failed to start connection thread");
        ctx->stop_requested = 1;
        pthread_join(ctx->rx_thread, NULL);
        close(ctx->socket_fd);
        pthread_mutex_destroy(&ctx->mutex);
        pthread_cond_destroy(&ctx->cond);
        free(ctx);
        return NULL;
    }

    if (!r3dmatch_wait_for_state(ctx, RCP_CONNECTION_STATE_CONNECTED, timeout_ms > 0 ? timeout_ms : R3DMATCH_CONN_TIMEOUT_MS))
    {
        if (ctx->last_error[0] == '\0')
        {
            snprintf(error_buf, error_buf_size, "RCP connection did not reach CONNECTED");
        }
        else
        {
            snprintf(error_buf, error_buf_size, "%s", ctx->last_error);
        }
        r3dmatch_rcp_close(ctx);
        return NULL;
    }

    snprintf(error_buf, error_buf_size, "");
    return ctx;
}

static int r3dmatch_rcp_get_supported_param(void *handle, rcp_param_t id)
{
    r3dmatch_rcp_ctx_t *ctx = (r3dmatch_rcp_ctx_t *)handle;
    if (!ctx || !ctx->con)
    {
        return 0;
    }
    return rcp_get_is_supported(ctx->con, id, NULL);
}

static int r3dmatch_rcp_set_parameter_int(void *handle, rcp_param_t id, int32_t value, const char *param_name, char *error_buf, size_t error_buf_size)
{
    r3dmatch_rcp_ctx_t *ctx = (r3dmatch_rcp_ctx_t *)handle;
    if (!ctx || !ctx->con)
    {
        snprintf(error_buf, error_buf_size, "invalid live RCP session");
        return 0;
    }
    if (!rcp_get_is_supported(ctx->con, id, NULL))
    {
        snprintf(error_buf, error_buf_size, "parameter not supported: %s", param_name);
        return 0;
    }
    if (!r3dmatch_set_param_int(ctx, id, value))
    {
        snprintf(error_buf, error_buf_size, "set failed for %s", param_name);
        return 0;
    }
    snprintf(error_buf, error_buf_size, "");
    return 1;
}

static int r3dmatch_rcp_get_parameter_int(void *handle, rcp_param_t id, const char *param_name, int timeout_ms, int32_t *out_value, char *error_buf, size_t error_buf_size)
{
    r3dmatch_rcp_ctx_t *ctx = (r3dmatch_rcp_ctx_t *)handle;
    if (!ctx || !ctx->con)
    {
        snprintf(error_buf, error_buf_size, "invalid live RCP session");
        return 0;
    }
    if (!rcp_get_is_supported(ctx->con, id, NULL))
    {
        snprintf(error_buf, error_buf_size, "parameter not supported: %s", param_name);
        return 0;
    }
    if (!r3dmatch_get_param_int(ctx, id, timeout_ms > 0 ? timeout_ms : R3DMATCH_CONN_TIMEOUT_MS, out_value))
    {
        if (ctx->last_error[0] != '\0')
        {
            snprintf(error_buf, error_buf_size, "%s", ctx->last_error);
        }
        else
        {
            snprintf(error_buf, error_buf_size, "timed out reading %s", param_name);
        }
        return 0;
    }
    snprintf(error_buf, error_buf_size, "");
    return 1;
}

int r3dmatch_rcp_get_supported_exposure_adjust(void *handle)
{
    return r3dmatch_rcp_get_supported_param(handle, RCP_PARAM_EXPOSURE_ADJUST);
}

int r3dmatch_rcp_get_supported_color_temperature(void *handle)
{
    return r3dmatch_rcp_get_supported_param(handle, RCP_PARAM_COLOR_TEMPERATURE);
}

int r3dmatch_rcp_get_supported_tint(void *handle)
{
    return r3dmatch_rcp_get_supported_param(handle, RCP_PARAM_TINT);
}

int r3dmatch_rcp_set_exposure_adjust(void *handle, int32_t value, char *error_buf, size_t error_buf_size)
{
    return r3dmatch_rcp_set_parameter_int(handle, RCP_PARAM_EXPOSURE_ADJUST, value, "exposure_adjust", error_buf, error_buf_size);
}

int r3dmatch_rcp_set_color_temperature(void *handle, int32_t value, char *error_buf, size_t error_buf_size)
{
    return r3dmatch_rcp_set_parameter_int(handle, RCP_PARAM_COLOR_TEMPERATURE, value, "color_temperature", error_buf, error_buf_size);
}

int r3dmatch_rcp_set_tint(void *handle, int32_t value, char *error_buf, size_t error_buf_size)
{
    return r3dmatch_rcp_set_parameter_int(handle, RCP_PARAM_TINT, value, "tint", error_buf, error_buf_size);
}

int r3dmatch_rcp_get_exposure_adjust(void *handle, int timeout_ms, int32_t *out_value, char *error_buf, size_t error_buf_size)
{
    return r3dmatch_rcp_get_parameter_int(handle, RCP_PARAM_EXPOSURE_ADJUST, "exposure_adjust", timeout_ms, out_value, error_buf, error_buf_size);
}

int r3dmatch_rcp_get_color_temperature(void *handle, int timeout_ms, int32_t *out_value, char *error_buf, size_t error_buf_size)
{
    return r3dmatch_rcp_get_parameter_int(handle, RCP_PARAM_COLOR_TEMPERATURE, "color_temperature", timeout_ms, out_value, error_buf, error_buf_size);
}

int r3dmatch_rcp_get_tint(void *handle, int timeout_ms, int32_t *out_value, char *error_buf, size_t error_buf_size)
{
    return r3dmatch_rcp_get_parameter_int(handle, RCP_PARAM_TINT, "tint", timeout_ms, out_value, error_buf, error_buf_size);
}

int r3dmatch_rcp_get_connection_state(void *handle)
{
    r3dmatch_rcp_ctx_t *ctx = (r3dmatch_rcp_ctx_t *)handle;
    if (!ctx)
    {
        return -1;
    }
    return ctx->connection_state_valid ? (int)ctx->connection_state : -1;
}

int r3dmatch_rcp_get_camera_info(void *handle, char *camera_id, size_t camera_id_size, char *camera_type, size_t camera_type_size, char *camera_version, size_t camera_version_size)
{
    r3dmatch_rcp_ctx_t *ctx = (r3dmatch_rcp_ctx_t *)handle;
    if (!ctx)
    {
        return 0;
    }
    if (camera_id && camera_id_size)
    {
        snprintf(camera_id, camera_id_size, "%s", ctx->camera_id);
    }
    if (camera_type && camera_type_size)
    {
        snprintf(camera_type, camera_type_size, "%s", ctx->camera_type);
    }
    if (camera_version && camera_version_size)
    {
        snprintf(camera_version, camera_version_size, "%s", ctx->camera_version);
    }
    return 1;
}

void r3dmatch_rcp_close(void *handle)
{
    r3dmatch_rcp_ctx_t *ctx = (r3dmatch_rcp_ctx_t *)handle;
    if (!ctx)
    {
        return;
    }
    ctx->stop_requested = 1;
    if (ctx->con)
    {
        rcp_delete_camera_connection(ctx->con);
        ctx->con = NULL;
    }
    shutdown(ctx->socket_fd, SHUT_RDWR);
    close(ctx->socket_fd);
    pthread_join(ctx->create_thread, NULL);
    pthread_join(ctx->rx_thread, NULL);
    pthread_cond_destroy(&ctx->cond);
    pthread_mutex_destroy(&ctx->mutex);
    free(ctx);
}
