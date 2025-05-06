#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

// 参考博客：https://blog.csdn.net/liuzheng1/article/details/90052050

// 一个路标点在某一图像帧上的信息
class FeaturePerFrame
{
  public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
    }
    double cur_td;
    Vector3d point;     // 路标点点空间坐标
    Vector2d uv;        // 路标点映射到该帧上的图像坐标
    Vector2d velocity;  // 路标点的跟踪速度
    double z;
    bool is_used;
    double parallax;
    MatrixXd A;
    VectorXd b;
    double dep_gradient;
};

// 一个路标点所有信息（每个路标点可以由多个连续的图像观测到）
class FeaturePerId
{
  public:
    const int feature_id;  // 路标点id
    int start_frame;       // 第一次出现该路标点的图像帧号
    vector<FeaturePerFrame> feature_per_frame;  // 包含该路标点的所有图像帧

    int used_num;  // 该路标点出现的次数
    bool is_outlier;  // 是否是外点
    bool is_margin;
    double estimated_depth;  // 逆深度
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail; 该特征点的状态，是否被三角化

    Vector3d gt_p;
    // 构造函数：以feature_id为索引，并保存了出现该路标点的第一帧的id
    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0)
    {
    }
    
    // 得到该路标点最后一次跟踪到的帧号
    int endFrame();
};

// 管理滑动窗口内所有路标点
class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    void debugShow();
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    void removeOutlier();
    list<FeaturePerId> feature;  // 包含滑动窗口内所有的路标点的链表
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif