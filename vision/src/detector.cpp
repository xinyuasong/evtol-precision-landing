#include "detector.hpp"
#include <opencv2/calib3d.hpp>

static std::vector<cv::Point3f> objectPoints(double s) {
    float h = (float)(s / 2);
    return {{-h, h, 0}, {h, h, 0}, {h, -h, 0}, {-h, -h, 0}};
}

TagDetector::TagDetector(const cv::Mat& K, const cv::Mat& dist, std::map<int, double> sizes, double reproj_max)
    : K_(K.clone()), dist_(dist.clone()), sizes_(std::move(sizes)), reproj_max_(reproj_max) {
    auto dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_APRILTAG_36h11);
    cv::aruco::DetectorParameters p;
    p.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    p.adaptiveThreshWinSizeMin = 3;
    p.adaptiveThreshWinSizeMax = 23;
    p.adaptiveThreshWinSizeStep = 10;
    p.minMarkerPerimeterRate = 0.02;
    det_ = cv::aruco::ArucoDetector(dict, p);
}

static double reproj(const std::vector<cv::Point3f>& obj, const std::vector<cv::Point2f>& img,
                     const cv::Vec3d& r, const cv::Vec3d& t, const cv::Mat& K, const cv::Mat& d) {
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj, r, t, K, d, proj);
    double e = 0;
    for (int i = 0; i < 4; ++i) e += cv::norm(proj[i] - img[i]);
    return e / 4.0;
}

std::vector<TagPoseWire> TagDetector::detect(const cv::Mat& gray, uint64_t t_us) {
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<int> ids;
    det_.detectMarkers(gray, corners, ids);
    std::vector<TagPoseWire> out;
    for (size_t i = 0; i < ids.size(); ++i) {
        auto it = sizes_.find(ids[i]);
        if (it == sizes_.end()) continue;
        auto obj = objectPoints(it->second);
        cv::Vec3d r, t;
        if (!cv::solvePnP(obj, corners[i], K_, dist_, r, t, false, cv::SOLVEPNP_IPPE_SQUARE)) continue;
        double err = reproj(obj, corners[i], r, t, K_, dist_);
        if (err > reproj_max_) {  // IPPE mirror case on fronto-parallel squares; see findings #6
            cv::Vec3d r2, t2;
            if (cv::solvePnP(obj, corners[i], K_, dist_, r2, t2, false, cv::SOLVEPNP_ITERATIVE)) {
                double e2 = reproj(obj, corners[i], r2, t2, K_, dist_);
                if (e2 < err) { r = r2; t = t2; err = e2; }
            }
        }
        TagPoseWire w{};
        w.t_capture_us = t_us; w.tag_id = ids[i];
        w.x = t[0]; w.y = t[1]; w.z = t[2];
        w.rx = r[0]; w.ry = r[1]; w.rz = r[2];
        w.reproj_err = err; w.valid = err < reproj_max_ ? 1 : 0; w.pad = 0;
        out.push_back(w);
    }
    return out;
}
