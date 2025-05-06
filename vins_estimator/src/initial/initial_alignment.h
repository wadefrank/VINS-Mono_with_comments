#pragma once
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <ros/ros.h>
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

/**
* @class        ImageFrame 图像帧
* @note         图像帧类可由图像帧的特征点与时间戳构造，
*               此外还保存了位姿R，t，预积分对象pre_integration，是否是关键帧。
*/
class ImageFrame
{
    public:
        ImageFrame(){};
        ImageFrame(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>>& _points, double _t):t{_t},is_key_frame{false}
        {
            points = _points;
        };
        map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>> > > points;  // 路标点信息
        double t;  // 时间戳
        Matrix3d R;  // 旋转
        Vector3d T;  // 平移
        IntegrationBase *pre_integration;  // 预积分对象
        bool is_key_frame;  // 是否是关键帧
};

bool VisualIMUAlignment(map<double, ImageFrame> &all_image_frame, Vector3d* Bgs, Vector3d &g, VectorXd &x);