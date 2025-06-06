#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "estimator.h"
#include "parameters.h"
#include "utility/visualization.h"


Estimator estimator;

std::condition_variable con;  // 条件变量
double current_time = -1;

queue<sensor_msgs::ImuConstPtr> imu_buf;
queue<sensor_msgs::PointCloudConstPtr> feature_buf;
queue<sensor_msgs::PointCloudConstPtr> relo_buf;    // 订阅pose graph node发布的回环帧数据，存到relo_buf队列中，供重定位使用

int sum_of_wait = 0;  // 图像帧等待imu帧的次数

// 互斥量
std::mutex m_buf;       // 用于处理多个线程使用imu_buf和feature_buf的冲突
std::mutex m_state;     // 用于处理多个线程使用当前里程计信息（即tmp_P、tmp_Q、tmp_V）的冲突
std::mutex i_buf;       
std::mutex m_estimator; // 用于处理多个线程使用VINS系统对象（即Estimator类的实例estimator）的冲突

double latest_time;     // 最近一次里程计信息对应的IMU时间戳

// IMU项[P, Q, B, Ba, Bg, a, g]
// 当前里程计信息
Eigen::Vector3d tmp_P;          // 平移（临时量）
Eigen::Quaterniond tmp_Q;       // 旋转（临时量）
Eigen::Vector3d tmp_V;          // 速度（临时量）

// 当前里程计信息对应的IMU bias
Eigen::Vector3d tmp_Ba;         // IMU加速度计偏置（临时量）
Eigen::Vector3d tmp_Bg;         // IMU陀螺仪偏置（临时量）

// 当前里程计信息对应的IMU测量值
Eigen::Vector3d acc_0;          // 上一帧IMU加速度测量值
Eigen::Vector3d gyr_0;          // 上一帧IMU角速度测量值

bool init_feature = 0;          // 0：第一次接收图像数据
bool init_imu = 1;              // 1：第一次接收IMU数据
double last_imu_t = 0;          // 上一帧IMU数据的时间戳（用于判断IMU数据时间是否正常，初始值为-1）

/**
 * @brief 基于IMU测量数据进行PVQ状态预测（位置、速度、姿态）
 * 
 * @details 本函数实现惯性导航的机械编排(Mechanical Alignment)，通过积分IMU的角速度和线加速度数据，
 *          更新载体的姿态、速度和位置预测值。通常用于组合导航或视觉惯性里程计(VIO)的预测步骤。
 * 
 * @param[in] imu_msg IMU消息，包含线加速度和角速度测量值（传感器坐标系下）
 */
void predict(const sensor_msgs::ImuConstPtr &imu_msg)
{
    // 1.处理IMU时间戳
    // 获取当前IMU时间戳（转换为秒）
    double t = imu_msg->header.stamp.toSec();

    // 记录第一帧IMU时间戳
    if (init_imu)
    {
        latest_time = t;
        init_imu = 0;
        return;
    }

    // 计算两次IMU测量的时间间隔（单位：秒）
    double dt = t - latest_time;

    // 更新上一帧IMU数据的时间戳
    latest_time = t;


    // 2.提取IMU测量值（传感器坐标系下）
    // 线加速度（m/s²）
    double dx = imu_msg->linear_acceleration.x;
    double dy = imu_msg->linear_acceleration.y;
    double dz = imu_msg->linear_acceleration.z;
    Eigen::Vector3d linear_acceleration{dx, dy, dz};

    // 角速度（rad/s）
    double rx = imu_msg->angular_velocity.x;
    double ry = imu_msg->angular_velocity.y;
    double rz = imu_msg->angular_velocity.z;
    Eigen::Vector3d angular_velocity{rx, ry, rz};


    // 3.状态更新
    // 计算校正后的上一帧IMU加速度
    Eigen::Vector3d un_acc_0 = tmp_Q * (acc_0 - tmp_Ba) - estimator.g;
    // 计算校正后的角速度（中值积分）
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - tmp_Bg;

    // 3.1 姿态更新：q_{k+1} = q_k ⊗ Δq(ω*dt)
    tmp_Q = tmp_Q * Utility::deltaQ(un_gyr * dt);

    // 计算校正后的当前IMU加速度
    Eigen::Vector3d un_acc_1 = tmp_Q * (linear_acceleration - tmp_Ba) - estimator.g;

    // 校正后的加速度中值积分
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);

    // 3.2 位置更新：P_{k+1} = P_k + V_k*dt + 0.5*a*dt²（二阶泰勒展开）
    tmp_P = tmp_P + dt * tmp_V + 0.5 * dt * dt * un_acc;

    // 3.3 速度更新：V_{k+1} = V_k + a*dt
    tmp_V = tmp_V + dt * un_acc;


    // 4.缓存当前IMU测量值用于下一帧计算
    acc_0 = linear_acceleration;    // 缓存当前加速度
    gyr_0 = angular_velocity;       // 缓存当前角速度
}

/**
 * @brief 从估计器中得到滑动窗口当前图像帧的imu更新项[P,Q,V,ba,bg,a,g]，对imu_buf中剩余的imu_msg进行PVQ递推
 * 
 * @note 当处理完measurements中的所有数据后，如果VINS系统正常完成滑动窗口优化，那么需要用优化后的结果更新里程计数据
 */
void update()
{
    TicToc t_predict;
    latest_time = current_time;

    // 首先获取滑动窗口中最新帧的P、V、Q
    tmp_P = estimator.Ps[WINDOW_SIZE];
    tmp_Q = estimator.Rs[WINDOW_SIZE];
    tmp_V = estimator.Vs[WINDOW_SIZE];
    tmp_Ba = estimator.Bas[WINDOW_SIZE];
    tmp_Bg = estimator.Bgs[WINDOW_SIZE];
    acc_0 = estimator.acc_0;
    gyr_0 = estimator.gyr_0;

    // 滑动窗口中最新帧并不是当前帧，中间隔着缓存队列的数据，所以还需要使用缓存队列中的IMU数据进行积分得到当前帧的里程计信息
    queue<sensor_msgs::ImuConstPtr> tmp_imu_buf = imu_buf;
    for (sensor_msgs::ImuConstPtr tmp_imu_msg; !tmp_imu_buf.empty(); tmp_imu_buf.pop())
        predict(tmp_imu_buf.front());

}

/**
 * @brief   对imu和图像数据进行对齐并组合
 * 
 * @note    img:    i -------- j  -  -------- k
 *          imu:    - jjjjjjjj - j/k kkkkkkkk -  
 *          直到把缓存中的图像特征数据或者IMU数据取完，才能够跳出此函数，并返回数据
 * 
 *           
 * @return  vector<std::pair<vector<ImuConstPtr>, PointCloudConstPtr>> (IMUs, img_msg)s
*/
std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>>
getMeasurements()
{
    std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

    // 直到把imu_buf或者feature_buf中的数据全部取出，才会退出while循环
    while (true)
    {
        if (imu_buf.empty() || feature_buf.empty())
            return measurements;

        // imu_buf队尾元素的时间戳，早于或等于feature_buf队首元素的时间戳（时间偏移补偿后），则需要等待接收IMU数据
        if (!(imu_buf.back()->header.stamp.toSec() > feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            //ROS_WARN("wait for imu, only should happen at the beginning");
            sum_of_wait++;
            return measurements;
        }

        // imu_buf队首元素的时间戳，晚于或等于feature_buf队首元素的时间戳（时间偏移补偿后），则需要剔除feature_buf队首多余的特征点数据
        if (!(imu_buf.front()->header.stamp.toSec() < feature_buf.front()->header.stamp.toSec() + estimator.td))
        {
            ROS_WARN("throw img, only should happen at the beginning");
            feature_buf.pop();
            continue;
        }
        sensor_msgs::PointCloudConstPtr img_msg = feature_buf.front();  // 读取feature_buf队首的数据
        feature_buf.pop();  // 剔除feature_buf队首的数据

        std::vector<sensor_msgs::ImuConstPtr> IMUs;

        // 一帧图像特征点数据，对应多帧imu数据,把它们进行对应，然后塞入measurements
        // 一帧图像特征点数据，与它和上一帧图像特征点数据之间的时间间隔内所有的IMU数据，以及时间戳晚于当前帧图像的第一帧IMU数据对应
        // 如下图所示：
        //  *             *             *             *             *            （IMU数据）
        //                                                    |                  （图像特征点数据）
        while (imu_buf.front()->header.stamp.toSec() < img_msg->header.stamp.toSec() + estimator.td)
        {
            IMUs.emplace_back(imu_buf.front());  
            imu_buf.pop();
        }
        IMUs.emplace_back(imu_buf.front());  // 时间戳晚于当前帧图像的第一帧IMU数据
        if (IMUs.empty())
            ROS_WARN("no imu between two image");
        measurements.emplace_back(IMUs, img_msg);
    }
    return measurements;
}

/**
 * @brief       IMU的回调函数，每接收到一个imu_msg就计算一次PVQ，并且封装成里程计信息发布
 * 
 * @details     订阅IMU数据，每订阅到一个IMU数据：
 *                  1.将IMU数据存入IMU数据缓存队列imu_buf
 *                  2.进行一次中值积分，计算在wolrd坐标系下的PVQ
 *                  3.并封装成Odometry信息发布
 * 
 * @param[in]   imu_msg
 * 
 * @return      void
 */
void imu_callback(const sensor_msgs::ImuConstPtr &imu_msg)
{
    if (imu_msg->header.stamp.toSec() <= last_imu_t)
    {
        ROS_WARN("imu message in disorder!");
        return;
    }
    last_imu_t = imu_msg->header.stamp.toSec();

    // 将IMU数据存入IMU数据缓存队列imu_buf
    m_buf.lock();
    imu_buf.push(imu_msg);
    m_buf.unlock();
    con.notify_one();  // 唤醒getMeasurements()读取缓存imu_buf和feature_buf中的观测数据

    // 通过IMU测量值积分更新并发布里程计信息
    last_imu_t = imu_msg->header.stamp.toSec();  // 这一行代码似乎重复了，上面有着一模一样的代码

    {
        std::lock_guard<std::mutex> lg(m_state);
        predict(imu_msg);
        std_msgs::Header header = imu_msg->header;
        header.frame_id = "world";
        // VINS初始化已完成，正处于滑动窗口非线性优化状态，如果VINS还在初始化，则不发布里程计信息
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            // 发布里程计信息，发布频率很高（与IMU数据同频），每次获取IMU数据都会及时进行更新，而且发布的是当前的里程计信息。
            pubLatestOdometry(tmp_P, tmp_Q, tmp_V, header);
            // 还有一个pubOdometry()函数，似乎也是发布里程计信息
            // 但是它是在estimator每次处理完一帧图像特征点数据后才发布的，有延迟，而且频率也不高（至多与图像同频）
    }
}

/**
 * @brief  图像特征回调函数，每订阅到一个图像帧，将图像特征点数据存入图像特征点数据缓存队列feature_buf
 * 
 * @param[in] feature_msg 
 */
void feature_callback(const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    if (!init_feature)
    {
        //skip the first detected feature, which doesn't contain optical flow speed
        init_feature = 1;
        return;
    }

    // 将图像特征点数据存入图像特征点数据缓存队列feature_buf
    m_buf.lock();
    feature_buf.push(feature_msg);
    m_buf.unlock();
    con.notify_one(); // 唤醒getMeasurements()读取缓存imu_buf和feature_buf中的观测数据
}

/**
 * @brief 判断是否重启estimator
 * 
 * @param restart_msg 
 */
void restart_callback(const std_msgs::BoolConstPtr &restart_msg)
{
    if (restart_msg->data == true)
    {
        ROS_WARN("restart the estimator!");
        m_buf.lock();
        while(!feature_buf.empty())
            feature_buf.pop();
        while(!imu_buf.empty())
            imu_buf.pop();
        m_buf.unlock();
        m_estimator.lock();
        estimator.clearState();
        estimator.setParameter();
        m_estimator.unlock();
        current_time = -1;
        last_imu_t = 0;
    }
    return;
}

/**
 * @brief 根据回环检测信息进行重定位
 * 
 * @param[in] points_msg 
 */
void relocalization_callback(const sensor_msgs::PointCloudConstPtr &points_msg)
{
    //printf("relocalization callback! \n");
    m_buf.lock();
    relo_buf.push(points_msg);
    m_buf.unlock();
}

// thread: visual-inertial odometry
/**
 * @brief       VIO后端，包括IMU预积分、松耦合初始化和local BA
 * 
 * @note        1.等待并且获取measurements：（IMUs, img_msg）s，计算dt
 *              2.estimator.processIMU() 进行IMU预积分
 *              3.estimator.setReloFrame() 设置重定位帧
 *              4.estimator.setprocessImage() 处理图像帧：初始化，紧耦合的非线性优化
 * 
 * @return      void
 */
void process()
{
    while (true)
    {
        
        std::vector<std::pair<std::vector<sensor_msgs::ImuConstPtr>, sensor_msgs::PointCloudConstPtr>> measurements;

        // 在执行getMeasurements()提取measurements时互斥锁m_buf会锁住，此时无法接收数据
        // getMeasurements()的作用是对imu和图像数据进行对齐并组合
        // unique_lock对象lk以独占所有权的方式管理mutex对象m_buf的上锁和解锁操作，所谓独占所有权，就是没有其他的 unique_lock对象同时拥有m_buf的所有权，
        // 新创建的unique_lock对象lk管理Mutex对象m_buf，并尝试调用m_buf.lock()对Mutex对象m_buf进行上锁，如果此时另外某个unique_lock对象已经管理了该Mutex对象m_buf,
        // 则当前线程将会被阻塞；如果此时m_buf本身就处于上锁状态，当前线程也会被阻塞（我猜的）。
        // 在unique_lock对象lk的声明周期内，它所管理的锁对象m_buf会一直保持上锁状态
        std::unique_lock<std::mutex> lk(m_buf);

        // std::condition_variable::wait(std::unique_lock<std::mutex>& lock, Predicate pred)的功能：
        // while (!pred()) 
        // {
        //     wait(lock);
        // }
        // 当pred为false的时候，才会调用wait(lock)，阻塞当前线程，当同一条件变量在其它线程中调用了notify_*函数时，当前线程被唤醒。
        // 直到pred为ture的时候，退出while循环。

        // [&]{return (measurements = getMeasurements()).size() != 0;}是lamda表达式（匿名函数）
        // 先调用匿名函数，从缓存队列中读取IMU数据和图像特征点数据，如果measurements为空，则匿名函数返回false，调用wait(lock)，
        // 释放m_buf（为了使图像和IMU回调函数可以访问缓存队列），阻塞当前线程，等待被con.notify_one()唤醒
        // 直到measurements不为空时（成功从缓存队列获取数据），匿名函数返回true，则可以退出while循环。
        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();

        m_estimator.lock();
        for (auto &measurement : measurements)
        {
            auto img_msg = measurement.second;
            double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;

            // 遍历该组 measurement 中的各帧imu数据，进行预积分
            for (auto &imu_msg : measurement.first)
            {
                double t = imu_msg->header.stamp.toSec();  // 最新IMU数据的时间戳
                double img_t = img_msg->header.stamp.toSec() + estimator.td;  // 图像特征点数据的时间戳，补偿了通过优化得到的一个时间偏移
                if (t <= img_t)  // 对于图像帧之前的所有IMU数据进行预积分
                { 
                    if (current_time < 0)  // 第一次接收IMU数据时会出现这种情况
                        current_time = t;
                    double dt = t - current_time;
                    ROS_ASSERT(dt >= 0);
                    current_time = t;  // 更新最近一次接收的IMU数据的时间戳

                    // IMU测量值数据
                    dx = imu_msg->linear_acceleration.x;
                    dy = imu_msg->linear_acceleration.y;
                    dz = imu_msg->linear_acceleration.z;
                    rx = imu_msg->angular_velocity.x;
                    ry = imu_msg->angular_velocity.y;
                    rz = imu_msg->angular_velocity.z;

                    // 预积分
                    estimator.processIMU(dt, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("imu: dt:%f a: %f %f %f w: %f %f %f\n",dt, dx, dy, dz, rx, ry, rz);

                }
                else
                {
                    double dt_1 = img_t - current_time;
                    double dt_2 = t - img_t;
                    current_time = img_t;
                    ROS_ASSERT(dt_1 >= 0);
                    ROS_ASSERT(dt_2 >= 0);
                    ROS_ASSERT(dt_1 + dt_2 > 0);
                    double w1 = dt_2 / (dt_1 + dt_2);
                    double w2 = dt_1 / (dt_1 + dt_2);
                    dx = w1 * dx + w2 * imu_msg->linear_acceleration.x;
                    dy = w1 * dy + w2 * imu_msg->linear_acceleration.y;
                    dz = w1 * dz + w2 * imu_msg->linear_acceleration.z;
                    rx = w1 * rx + w2 * imu_msg->angular_velocity.x;
                    ry = w1 * ry + w2 * imu_msg->angular_velocity.y;
                    rz = w1 * rz + w2 * imu_msg->angular_velocity.z;
                    estimator.processIMU(dt_1, Vector3d(dx, dy, dz), Vector3d(rx, ry, rz));
                    //printf("dimu: dt:%f a: %f %f %f w: %f %f %f\n",dt_1, dx, dy, dz, rx, ry, rz);
                }
            }

            // 设置重定位用的回环帧
            // set relocalization frame
            sensor_msgs::PointCloudConstPtr relo_msg = NULL;

            while (!relo_buf.empty())
            {
                relo_msg = relo_buf.front();
                relo_buf.pop();
            }
            if (relo_msg != NULL)
            {
                vector<Vector3d> match_points;
                double frame_stamp = relo_msg->header.stamp.toSec();  // 回环帧的时间戳
                for (unsigned int i = 0; i < relo_msg->points.size(); i++)
                {
                    Vector3d u_v_id;
                    u_v_id.x() = relo_msg->points[i].x;
                    u_v_id.y() = relo_msg->points[i].y;
                    u_v_id.z() = relo_msg->points[i].z;
                    match_points.push_back(u_v_id);
                }
                Vector3d relo_t(relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
                Quaterniond relo_q(relo_msg->channels[0].values[3], relo_msg->channels[0].values[4], relo_msg->channels[0].values[5], relo_msg->channels[0].values[6]);
                Matrix3d relo_r = relo_q.toRotationMatrix();
                int frame_index;
                frame_index = relo_msg->channels[0].values[7];
                estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);  // 设置回环帧
            }

            ROS_DEBUG("processing vision data with stamp %f \n", img_msg->header.stamp.toSec());

            TicToc t_s;

            // image为字典，用来存储单帧所有特征点的信息，键为feature_id，值为vector<pair<camera_id,[x,y,z,u,v,vx,vy]>>
            // 推测：采用上述结构表示可能是因为一个特征点可能在多个相机中出现（双目），但是在VINS-Mono中camera_id=0
            map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> image;
            for (unsigned int i = 0; i < img_msg->points.size(); i++)
            {
                // v表示特征点id（+0.5表示四舍五入）
                int v = img_msg->channels[0].values[i] + 0.5;

                // 特征点id计算方式： id_of_point = p_id * NUM_OF_CAM + i
                int feature_id = v / NUM_OF_CAM;
                int camera_id = v % NUM_OF_CAM;

                // points[i]表示特征点在归一化相机坐标系下的坐标（z=1）
                double x = img_msg->points[i].x;
                double y = img_msg->points[i].y;
                double z = img_msg->points[i].z;

                // p_u和p_v分别表示特征点的像素坐标x,y
                double p_u = img_msg->channels[1].values[i];
                double p_v = img_msg->channels[2].values[i];

                // velocity_x和velocity_y分别表示特征点沿x,y方向的像素移动速度
                double velocity_x = img_msg->channels[3].values[i];
                double velocity_y = img_msg->channels[4].values[i];


                ROS_ASSERT(z == 1);
                Eigen::Matrix<double, 7, 1> xyz_uv_velocity;
                xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y;
                image[feature_id].emplace_back(camera_id,  xyz_uv_velocity);
            }

            // 处理图像特征：包括初始化和非线性优化
            estimator.processImage(image, img_msg->header);

            double whole_t = t_s.toc();
            printStatistics(estimator, whole_t);
            std_msgs::Header header = img_msg->header;
            header.frame_id = "world";

            pubOdometry(estimator, header);
            pubKeyPoses(estimator, header);
            pubCameraPose(estimator, header);
            pubPointCloud(estimator, header);
            pubTF(estimator, header);
            pubKeyframe(estimator);
            if (relo_msg != NULL)
                pubRelocalization(estimator);
            //ROS_ERROR("end: %f, at %f", img_msg->header.stamp.toSec(), ros::Time::now().toSec());
        }
        m_estimator.unlock();
        m_buf.lock();
        m_state.lock();
        if (estimator.solver_flag == Estimator::SolverFlag::NON_LINEAR)
            update();
        m_state.unlock();
        m_buf.unlock();
    }
}

int main(int argc, char **argv)
{
    // ROS初始化
    ros::init(argc, argv, "vins_estimator");

    // 设置句柄
    ros::NodeHandle n("~");
    
    // 设置logger的级别，只有级别大于或者等于level（这里是Info）的日志消息才会得到处理
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    // 读取yaml配置文件中的一些配置参数
    readParameters(n);

    // 设置估计器参数
    estimator.setParameter();

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    ROS_WARN("waiting for image and imu...");

    // rviz相关话题
    registerPub(n);  // 注册visualization.cpp中创建的发布器

    // 订阅IMU数据
    ros::Subscriber sub_imu = n.subscribe(IMU_TOPIC, 2000, imu_callback, ros::TransportHints().tcpNoDelay());

    // 订阅图像特征点数据
    ros::Subscriber sub_image = n.subscribe("/feature_tracker/feature", 2000, feature_callback);

    // 判断是否重启estimator
    ros::Subscriber sub_restart = n.subscribe("/feature_tracker/restart", 2000, restart_callback);

    // 根据回环检测信息进行重定位
    ros::Subscriber sub_relo_points = n.subscribe("/pose_graph/match_points", 2000, relocalization_callback);

    // measurement_process线程的线程函数是process()，在process()中处理VIO后端，包括IMU预积分、松耦合初始化和local BA
    std::thread measurement_process{process};
    ros::spin();

    return 0;
}
