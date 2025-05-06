#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "feature_tracker.h"

#define SHOW_UNDISTORTION 0

vector<uchar> r_status;
vector<float> r_err;
queue<sensor_msgs::ImageConstPtr> img_buf;

ros::Publisher pub_img,pub_match;
ros::Publisher pub_restart;

FeatureTracker trackerData[NUM_OF_CAM];     //每个相机都有一个FeatureTracker实例，即trackerData[i]
double first_image_time;
int pub_count = 1;                          // 发布计数
bool first_image_flag = true;               // 用于判断是否是第一帧（是为true，不是为false）
double last_image_time = 0;                 // 用于记录上一帧图像的时间
bool init_pub = 0;                          // 用于判断是否发布特征点（因为第一帧没有光流，所以不发布）

/**
 * @brief       回调函数，对新接收到的图像进行特征点的追踪，发布
 * 
 * @details     readImage()函数对新来的图像使用光流法进行特征点跟踪
 *              追踪的特征点封装成feature_points发布到pub_img的话题下
 *              图像封装成ptr发布在pub_match下
 * 
 * @param[in]   img_msg     输入的图像
 * 
 * @return      void
 */
void img_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    // 判断是否是第一帧
    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = img_msg->header.stamp.toSec();       // 记录第一个图像帧的时间
        last_image_time = img_msg->header.stamp.toSec();        // 同时记录上一帧图像的时间
        return;
    }

    // detect unstable camera stream
    // 如果观测到不稳定的相机数据流，则restart (通过判断不同相机帧的时间戳)
    if (img_msg->header.stamp.toSec() - last_image_time > 1.0 || img_msg->header.stamp.toSec() < last_image_time)
    {
        ROS_WARN("image discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }

    last_image_time = img_msg->header.stamp.toSec();        // 当不是第一帧时，记录上一帧的时间

    // frequency control
    // 发布频率控制 (并不是每读入一帧图像，就要发布特征点)
    // 判断当前接收到图像的频率，并和设定的频率FREQ进行对比 (round() C函数：四舍五入)
    // 小于则发布特征点，否则不发布 (euroc数据中FRWQ = 10)
    if (round(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        // reset the frequency control
        // 时间间隔内的发布频率十分接近设定频率时，更新时间间隔起始时刻，并将数据发布次数置0
        if (abs(1.0 * pub_count / (img_msg->header.stamp.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = img_msg->header.stamp.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    cv_bridge::CvImageConstPtr ptr;

    // 将图像编码8UC1转换为mono8，即存储下来的图像为单色，8Bit的图片，一般是bmp，jpeg等
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        // 构建CV:Mat与sensor_masg::Image之间的桥梁。
        // 参考http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages 
        // 注意，img_msg或img都是sensor_msg格式的，我们需要一个桥梁，转换为CV::Mat格式的数据，以供后续图像处理。
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    
    cv::Mat show_img = ptr->image;
    TicToc t_r;

    // 对最新帧进行特征点的提取和光流追踪(img_callback()的核心语句)
    for (int i = 0; i < NUM_OF_CAM; i++)
    {
        ROS_DEBUG("processing camera %d", i);
        
        // 单目情况下
        if (i != 1 || !STEREO_TRACK)
            // 使用readImage()函数读取图像数据进行处理,进行特征点的提取和光流追踪
            trackerData[i].readImage(ptr->image.rowRange(ROW * i, ROW * (i + 1)), img_msg->header.stamp.toSec());
        // 双目情况下，因为是VINS-Mono，所以暂时不用看
        else
        {
            // 判断是否对图像进行自适应直方图均衡化
            if (EQUALIZE)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
                clahe->apply(ptr->image.rowRange(ROW * i, ROW * (i + 1)), trackerData[i].cur_img);
            }
            else
                trackerData[i].cur_img = ptr->image.rowRange(ROW * i, ROW * (i + 1));
        }

#if SHOW_UNDISTORTION
        trackerData[i].showUndistortion("undistrotion_" + std::to_string(i));
#endif
    }

    // 对新加入的特征点更新全局id
    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                // completed(或者是update())如果是true，说明没有更新完id，则持续循环，
                // 如果是false，说明更新完了则跳出循环。
                // 注意n_id是static类型的数据，具有累加的功能。
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

    // 1、将特征点id，矫正后归一化平面的3D点(x,y,z=1)，像素2D点(u,v)，像素的速度(vx,vy)，
    // 封装成sensor_msgs::PointCloudPtr类型的feature_points实例中,发布到pub_img;
    // 2、将图像封装到cv_bridge::cvtColor类型的ptr实例中发布到pub_match
   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = img_msg->header;
        feature_points->header.frame_id = "world";

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;

                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }
        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        
        // skip the first image; since no optical speed on frist image
        // 第一帧不发布，因为没有光流速度
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);

        if (SHOW_TRACK)
        {
            ptr = cv_bridge::cvtColor(ptr, sensor_msgs::image_encodings::BGR8);
            //cv::Mat stereo_img(ROW * NUM_OF_CAM, COL, CV_8UC3);
            cv::Mat stereo_img = ptr->image;

            for (int i = 0; i < NUM_OF_CAM; i++)
            {
                cv::Mat tmp_img = stereo_img.rowRange(i * ROW, (i + 1) * ROW);
                cv::cvtColor(show_img, tmp_img, CV_GRAY2RGB);

                // 显示追踪状态，越红越好，越蓝越不行
                for (unsigned int j = 0; j < trackerData[i].cur_pts.size(); j++)
                {
                    double len = std::min(1.0, 1.0 * trackerData[i].track_cnt[j] / WINDOW_SIZE);
                    cv::circle(tmp_img, trackerData[i].cur_pts[j], 2, cv::Scalar(255 * (1 - len), 0, 255 * len), 2);
                    //draw speed line
                    /*
                    Vector2d tmp_cur_un_pts (trackerData[i].cur_un_pts[j].x, trackerData[i].cur_un_pts[j].y);
                    Vector2d tmp_pts_velocity (trackerData[i].pts_velocity[j].x, trackerData[i].pts_velocity[j].y);
                    Vector3d tmp_prev_un_pts;
                    tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                    tmp_prev_un_pts.z() = 1;
                    Vector2d tmp_prev_uv;
                    trackerData[i].m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
                    cv::line(tmp_img, trackerData[i].cur_pts[j], cv::Point2f(tmp_prev_uv.x(), tmp_prev_uv.y()), cv::Scalar(255 , 0, 0), 1 , 8, 0);
                    */
                    //char name[10];
                    //sprintf(name, "%d", trackerData[i].ids[j]);
                    //cv::putText(tmp_img, name, trackerData[i].cur_pts[j], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
                }
            }
            //cv::imshow("vis", stereo_img);
            //cv::waitKey(5);
            pub_match.publish(ptr->toImageMsg());
        }
    }
    ROS_INFO("whole feature tracker processing costs: %f", t_r.toc());
}

int main(int argc, char **argv)
{
    // ros初始化和设置句柄
    ros::init(argc, argv, "feature_tracker");

    // 设置句柄
    ros::NodeHandle n("~");

    // 设置logger的级别，只有级别大于或者等于level（这里是Info）的日志消息才会得到处理
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    
    // 读取yaml配置文件中的一些配置参数
    readParameters(n);

    // 读取每个相机实例对应的相机内参
    for (int i = 0; i < NUM_OF_CAM; i++)
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]);

    // 判断是否加入鱼眼mask来去除边缘噪声
    if(FISHEYE)
    {
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            trackerData[i].fisheye_mask = cv::imread(FISHEYE_MASK, 0);
            if(!trackerData[i].fisheye_mask.data)
            {
                ROS_INFO("load mask fail");
                ROS_BREAK();
            }
            else
                ROS_INFO("load mask success");
        }
    }

    // 订阅话题IMAGE_TOPIC(/cam0/image_raw),执行回调函数img_callback
    ros::Subscriber sub_img = n.subscribe(IMAGE_TOPIC, 100, img_callback);

    // 发布feature，实例feature_point，跟踪的特征点，给后端优化用
    pub_img = n.advertise<sensor_msgs::PointCloud>("feature", 1000);

    // 发布feature_img，实例ptr，跟踪的特征点图，给RVIZ用和调试用
    pub_match = n.advertise<sensor_msgs::Image>("feature_img",1000);
    
    // 发布restart
    pub_restart = n.advertise<std_msgs::Bool>("restart",1000);

    /*
    if (SHOW_TRACK)
        cv::namedWindow("vis", cv::WINDOW_NORMAL);
    */

    ros::spin();
    return 0;
}


// new points velocity is 0, pub or not?
// track cnt > 1 pub?