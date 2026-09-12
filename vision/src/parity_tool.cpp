// Detect on one PNG and print poses as JSON — used by tests/test_detector_parity.py.
#include "detector.hpp"
#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>
#include <cstdio>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: parity_tool image.png calib.yaml tags.yaml\n"); return 2; }
    cv::Mat gray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    YAML::Node c = YAML::LoadFile(argv[2]);
    cv::Mat K(3, 3, CV_64F), dist(1, 5, CV_64F);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) K.at<double>(i, j) = c["camera_matrix"][i][j].as<double>();
    for (int i = 0; i < 5; ++i) dist.at<double>(0, i) = c["distortion_coefficients"][i].as<double>();
    std::map<int, double> sizes;
    for (auto t : YAML::LoadFile(argv[3])["tags"]) sizes[t["id"].as<int>()] = t["size_m"].as<double>();
    TagDetector det(K, dist, sizes);
    std::printf("[");
    bool first = true;
    for (const auto& p : det.detect(gray, 0)) {
        std::printf("%s{\"tag_id\":%d,\"x\":%.9f,\"y\":%.9f,\"z\":%.9f,\"rx\":%.9f,\"ry\":%.9f,\"rz\":%.9f,\"reproj_err\":%.6f,\"valid\":%d}",
                    first ? "" : ",", p.tag_id, p.x, p.y, p.z, p.rx, p.ry, p.rz, p.reproj_err, p.valid);
        first = false;
    }
    std::printf("]\n");
    return 0;
}
