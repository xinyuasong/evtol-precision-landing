#include "frame_source.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

static uint64_t now_us() {
    using namespace std::chrono;
    return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

namespace {

struct CameraSource : FrameSource {
    cv::VideoCapture cap;
    CameraSource(int dev, int w, int h, int fps) : cap(dev, cv::CAP_V4L2) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, w);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, h);
        cap.set(cv::CAP_PROP_FPS, fps);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);  // never queue frames: newest only
        if (!cap.isOpened()) throw std::runtime_error("camera open failed");
    }
    bool next(cv::Mat& gray, uint64_t& t) override {
        cv::Mat bgr;
        if (!cap.read(bgr)) return false;
        t = now_us();  // timestamp at capture, before any processing
        if (bgr.channels() == 1) gray = bgr; else cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
        return true;
    }
};

struct VideoSource : FrameSource {
    cv::VideoCapture cap;
    explicit VideoSource(const std::string& p) : cap(p) {
        if (!cap.isOpened()) throw std::runtime_error("video open failed: " + p);
    }
    bool next(cv::Mat& gray, uint64_t& t) override {
        cv::Mat f;
        if (!cap.read(f)) return false;
        t = now_us();
        if (f.channels() == 1) gray = f; else cv::cvtColor(f, gray, cv::COLOR_BGR2GRAY);
        return true;
    }
};

// Protocol: the simulator connects and sends frames as
//   u64 t_capture_us | u32 width | u32 height | width*height bytes (8-bit gray)
struct MockSource : FrameSource {
    int listen_fd = -1, fd = -1;
    MockSource(const std::string& host, int port) {
        listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(port);
        ::inet_pton(AF_INET, host.c_str(), &a.sin_addr);
        if (::bind(listen_fd, (sockaddr*)&a, sizeof a) < 0 || ::listen(listen_fd, 1) < 0)
            throw std::runtime_error("mock source bind/listen failed");
        fd = ::accept(listen_fd, nullptr, nullptr);
        if (fd < 0) throw std::runtime_error("mock source accept failed");
    }
    ~MockSource() override { if (fd >= 0) ::close(fd); if (listen_fd >= 0) ::close(listen_fd); }
    bool readAll(void* buf, size_t n) {
        auto* p = static_cast<char*>(buf);
        while (n) {
            ssize_t r = ::read(fd, p, n);
            if (r <= 0) return false;
            p += r; n -= (size_t)r;
        }
        return true;
    }
    bool next(cv::Mat& gray, uint64_t& t) override {
        uint64_t ts; uint32_t w, h;
        if (!readAll(&ts, 8) || !readAll(&w, 4) || !readAll(&h, 4)) return false;
        gray.create((int)h, (int)w, CV_8UC1);
        if (!readAll(gray.data, (size_t)w * h)) return false;
        t = ts;
        return true;
    }
};

}  // namespace

std::unique_ptr<FrameSource> makeCameraSource(int d, int w, int h, int fps) { return std::make_unique<CameraSource>(d, w, h, fps); }
std::unique_ptr<FrameSource> makeMockSource(const std::string& host, int port) { return std::make_unique<MockSource>(host, port); }
std::unique_ptr<FrameSource> makeVideoSource(const std::string& p) { return std::make_unique<VideoSource>(p); }
