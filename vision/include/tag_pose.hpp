// Wire struct shared with control/pose_sub.py (WIRE_FMT "<QidddddddBx", 70 bytes, little-endian).
#pragma once
#include <cstdint>
#pragma pack(push, 1)
struct TagPoseWire {
    uint64_t t_capture_us;
    int32_t  tag_id;
    double   x, y, z;        // tag origin in camera frame, metres
    double   rx, ry, rz;     // Rodrigues rotation, tag -> camera
    double   reproj_err;     // px
    uint8_t  valid;
    uint8_t  pad;
};
#pragma pack(pop)
static_assert(sizeof(TagPoseWire) == 70, "wire struct must match control/pose_sub.py");
