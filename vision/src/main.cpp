// Vision node: capture -> detect -> publish TagPose over UDP to the control node.
//   vision_node --source camera|mock|video [--device 0] [--video f.mp4] [--calib calib/camera_640x480.yaml]
//               [--tags config/tags.yaml] [--udp-host 127.0.0.1] [--udp-port 5600] [--mock-port 5700]
#include "detector.hpp"
#include "frame_source.hpp"
#include <opencv2/core/persistence.hpp>
#include <yaml-cpp/yaml.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>
#include <iostream>
#include <string>

static std::string arg(int argc, char** argv, const std::string& k, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) if (k == argv[i]) return argv[i + 1];
    return def;
}

int main(int argc, char** argv) {
    std::string source = arg(argc, argv, "--source", "mock");
    std::string calib = arg(argc, argv, "--calib", "calib/camera_640x480.yaml");
    std::string tags = arg(argc, argv, "--tags", "config/tags.yaml");
    std::string host = arg(argc, argv, "--udp-host", "127.0.0.1");
    int port = std::stoi(arg(argc, argv, "--udp-port", "5600"));

    YAML::Node c = YAML::LoadFile(calib);
    cv::Mat K(3, 3, CV_64F), dist(1, 5, CV_64F);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) K.at<double>(i, j) = c["camera_matrix"][i][j].as<double>();
    for (int i = 0; i < 5; ++i) dist.at<double>(0, i) = c["distortion_coefficients"][i].as<double>();
    std::map<int, double> sizes;
    for (auto t : YAML::LoadFile(tags)["tags"]) sizes[t["id"].as<int>()] = t["size_m"].as<double>();

    std::unique_ptr<FrameSource> src;
    if (source == "camera") src = makeCameraSource(std::stoi(arg(argc, argv, "--device", "0")), 640, 480, 30);
    else if (source == "video") src = makeVideoSource(arg(argc, argv, "--video", ""));
    else src = makeMockSource("127.0.0.1", std::stoi(arg(argc, argv, "--mock-port", "5700")));

    int fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    sockaddr_in dst{};
    dst.sin_family = AF_INET; dst.sin_port = htons(port);
    ::inet_pton(AF_INET, host.c_str(), &dst.sin_addr);

    TagDetector det(K, dist, sizes);
    cv::Mat gray; uint64_t t;
    while (src->next(gray, t)) {
        for (const auto& p : det.detect(gray, t))
            ::sendto(fd, &p, sizeof p, 0, (sockaddr*)&dst, sizeof dst);
    }
    ::close(fd);
    return 0;
}
