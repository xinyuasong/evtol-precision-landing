#pragma once
#include <opencv2/objdetect/aruco_detector.hpp>
#include <map>
#include <vector>
#include "tag_pose.hpp"

// AprilTag 36h11 detection + planar PnP, mirroring vision/py/detector_py.py exactly:
// same corner order, IPPE_SQUARE with an ITERATIVE fallback when reprojection fails,
// SUBPIX corner refinement, same reprojection gate.
class TagDetector {
public:
    TagDetector(const cv::Mat& K, const cv::Mat& dist, std::map<int, double> tag_sizes, double reproj_max_px = 2.0);
    std::vector<TagPoseWire> detect(const cv::Mat& gray, uint64_t t_capture_us);
private:
    cv::Mat K_, dist_;
    std::map<int, double> sizes_;
    double reproj_max_;
    cv::aruco::ArucoDetector det_;
};
