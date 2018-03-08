#include "iostream"
#include "ros/ros.h"
#include "pcl/filters/crop_box.h"
#include "sensor_msgs/PointCloud2.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>

#include "object_detector/Box.h"

bool crop_pointcloud(object_detector::Box::Request &req, object_detector::Box::Response &res){
    std::cout << req.x << std::endl;
    std::cout << req.y << std::endl;
    std::cout << req.width << std::endl;
    std::cout << req.height << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut (new pcl::PointCloud<pcl::PointXYZ>);

//    pcl::PCLPointCloud2 temp_conversion;
//    pcl_conversions::toPCL(req.whole_pointcloud, temp_conversion);
//    pcl::fromPCLPointCloud2(temp_conversion, *cloudIn);
    pcl::fromROSMsg(req.whole_pointcloud, *cloudIn);

    pcl::CropBox<pcl::PointXYZ> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(req.x - req.width/2, req.y - req.height/2, req.z, 1.0)); // TODO find median z
    boxFilter.setMax(Eigen::Vector4f(req.x + req.width/2, req.y + req.height/2, req.z, 1.0)); // TODO find median z
    boxFilter.setInputCloud(cloudIn);
    pcl::PointCloud<pcl::PointXYZ>::Ptr bodyFiltered (new pcl::PointCloud<pcl::PointXYZ>);
    boxFilter.filter(*cloudOut);
    pcl::toROSMsg(*cloudOut, res.object_pointcloud);

    return true;
}

int main(int argc, char **argv){
    ros::init(argc, argv, "pointcloud_cropper");
    ros::NodeHandle n;

    ros::ServiceServer service = n.advertiseService("crop_pointcloud", crop_pointcloud);
    ROS_INFO("Ready to crop pointclouds like a god damn ninja.");
    ros::spin();
    return 0;
}