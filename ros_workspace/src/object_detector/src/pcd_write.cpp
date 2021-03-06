#include "../include/object_detector/pcd_write.hpp"


Eigen::VectorXf pfh_function(pcl::PCLPointCloud2::Ptr init_cloud){
    /////////////////// Apply Voxel Grid Filtering //////////////////////////
    pcl::PCLPointCloud2::Ptr cloud_filtered(new pcl::PCLPointCloud2 ());
    std::cout << "PointCloud before filtering: " << init_cloud->width * init_cloud->height
       << " data points (" << pcl::getFieldsList (*init_cloud) << "). \n";

    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud(init_cloud);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(*cloud_filtered);

    std::cout << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
       << " data points (" << pcl::getFieldsList (*cloud_filtered) << "). \n";
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromPCLPointCloud2(*cloud_filtered, *cloud);

    ///////////////////////////// Estimate the normals //////////////////////////////////////////////
    // Create the normal estimation class, and pass the input dataset to it
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    //TODO take it from the pointcloud - it is the same now anyway.
    normalEstimation.setViewPoint(0, 0, 0);
    normalEstimation.setInputCloud(cloud);

    // Create an empty kdtree representation, and pass it to the normal estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>());
    normalEstimation.setSearchMethod(kdtree);

    // Output datasets
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

    // Use all neighbors in a sphere of radius 3cm
    normalEstimation.setRadiusSearch(0.03);

    // Compute the features
    normalEstimation.compute(*cloud_normals);
    // cloud_normals->points.size (); should have the same size as the input cloud->points.size ()*

    for (int i = 0; i < cloud_normals->points.size(); i++){
        if (!pcl::isFinite<pcl::Normal>(cloud_normals->points[i])){
            // TODO handle this
            PCL_WARN("normals[%d] is not finite\n", i);
        }
    }
    std::cout << "Normal Estimation is done. \n";

    //////////// Create the PFH estimation class, and pass the input dataset+normals to it /////////////////
    pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
    int n_sub_div = 5;
    Eigen::VectorXf pfh_histogram(n_sub_div * n_sub_div * n_sub_div);
    int size_of_indices = cloud_filtered->width * cloud_filtered->height;
    std::vector<int> indices;
    for (int i = 0; i < size_of_indices; i++){
        indices.push_back(i);
    }
    pfh.computePointPFHSignature(*cloud, *cloud_normals, indices, n_sub_div, pfh_histogram);
    std::cout << "Point Feature Histograms are ready." << endl;

    /////////////////////////////////// Visualisation ////////////////////////
    // Visualize the Filtered PointCloud
//    pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
//    viewer.showCloud(cloud);
//    while (!viewer.wasStopped ())
//    {
//    }
    return pfh_histogram;
}

int pfh_estimator(std::string object_name, std::string id1, std::string id2, std::string id3, ofstream &write_file,
    int object_id){

    std::string file_location = "Database/PCL_dataset/" + object_name + "/" + object_name + "_" + id1
        + "/" + object_name + "_" + id1 + "_" + id2 + "_" + id3 + ".pcd";
    std::cout << "Doing stuff for object at location: " << file_location << endl;
    pcl::PCLPointCloud2::Ptr init_cloud(new pcl::PCLPointCloud2 ());
    pcl::PCDReader reader;
    if (reader.read (file_location, *init_cloud) == -1){
        PCL_ERROR ("Couldn't read file \n");
        return -1;
    }

    Eigen::VectorXf pfh_histogram;
    pfh_histogram = pfh_function(init_cloud);

    // Write the data in a csv file.
    write_file << object_name + "_" + id1 + "_" + id2 + "_" + id3 << ",";
    for(int i = 0; i < 125; i++){
        write_file << pfh_histogram[i] << ",";
    }
    write_file << object_id << "\n";
    return 1;
}

Eigen::VectorXf pfh_estimator(pcl::PointCloud<pcl::PointXYZ> input_cloud){
    Eigen::VectorXf pfh_histogram;
    pcl::PCLPointCloud2::Ptr point_cloud2(new pcl::PCLPointCloud2());

    pcl::toPCLPointCloud2(input_cloud, *point_cloud2);
    pfh_histogram = pfh_function(point_cloud2);

    return pfh_histogram;
}
