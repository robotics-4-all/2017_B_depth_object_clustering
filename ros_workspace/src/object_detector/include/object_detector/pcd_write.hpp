#ifndef PCD_WRITE_HPP_  // To make sure you don't declare the function
#define PCD_WRITE_HPP_  // more than once by including the header multiple times.

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/pfh.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/visualization/cloud_viewer.h>
#include <fstream>
#include <string>


Eigen::VectorXf pfh_function(pcl::PCLPointCloud2::Ptr init_cloud);

int pfh_estimator(std::string object_name, std::string id1, std::string id2, std::string id3, ofstream &write_file,
    int object_id);

int pfh_estimator(pcl::PointCloud<pcl::PointXYZ> input_cloud);

#endif