#include "iostream"
#include "ros/ros.h"
#include "pcl/filters/crop_box.h"
#include "sensor_msgs/PointCloud2.h"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl/point_types.h"
#include "pcl/PCLPointCloud2.h"
#include "pcl/conversions.h"

#include "object_detector/Box.h"
#include "../include/object_detector/pcd_write.hpp"


// Service function
bool crop_pointcloud(object_detector::Box::Request &req, object_detector::Box::Response &res){

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudIn(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(req.whole_pointcloud, *cloudIn);

    pcl::CropBox<pcl::PointXYZ> boxFilter;
    boxFilter.setMin(Eigen::Vector4f(req.x - req.width/2, req.y - req.height/2, req.z - 1, 1.0)); // TODO find overlap
                                                                                                  // TODO of objects
    boxFilter.setMax(Eigen::Vector4f(req.x + req.width/2, req.y + req.height/2, req.z + 1, 1.0)); // TODO find median z
    boxFilter.setInputCloud(cloudIn);
    pcl::PointCloud<pcl::PointXYZ>::Ptr bodyFiltered(new pcl::PointCloud<pcl::PointXYZ>);
    boxFilter.filter(*cloudOut);

    // Visualization
//    pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
//    viewer.showCloud(cloudOut);
//    while (!viewer.wasStopped())
//    {
//    }

    // Convert from pcl::PointCloud<pcl::PointXYZ> to sensor_msgs/PointCloud2 and send the response back to client.
//    pcl::toROSMsg(*cloudOut, res.object_pointcloud);
//    std::cout << "My stuff is " << pfh_estimator(*cloudOut) << std::endl;
    Eigen::VectorXf temp_vector = pfh_estimator(*cloudOut);
    for (int i = 0; i<125; i++){
        res.pfh[i] = temp_vector[i];
//        std::cout << temp_vector << std::endl;
    }

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