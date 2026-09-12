#pragma once
#include <opencv2/core.hpp>
#include <cstdint>
#include <memory>
#include <string>

// A source of grayscale frames with a capture timestamp (microseconds, monotonic).
struct FrameSource {
    virtual ~FrameSource() = default;
    virtual bool next(cv::Mat& gray, uint64_t& t_capture_us) = 0;
};

// Real camera (V4L2 via OpenCV). Written for hardware; not runnable in CI.
std::unique_ptr<FrameSource> makeCameraSource(int device, int width, int height, int fps);
// Frames pushed by the simulator over a local TCP socket as raw 8-bit images.
std::unique_ptr<FrameSource> makeMockSource(const std::string& host, int port);
// Frames from a video file (replay).
std::unique_ptr<FrameSource> makeVideoSource(const std::string& path);
