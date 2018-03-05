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


int find_a_name_later(std::string object_name, std::string id1, std::string id2, std::string id3, ofstream &write_file){

    std::string file_location = "Database/PCL_dataset/" + object_name + "_" + id1 + "/" + object_name + "_" + id1 + "_" + id2 + "_" + id3 + ".pcd";
    std::cout << "Doing stuff for object at location: " << file_location << endl;
    pcl::PCLPointCloud2::Ptr init_cloud(new pcl::PCLPointCloud2 ());
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PCDReader reader;
    if (reader.read (file_location, *init_cloud) == -1){
        PCL_ERROR ("Couldn't read file \n");
        return -1;
    }

    /////////////////// Apply Voxel Grid Filtering //////////////////////////
    pcl::PCLPointCloud2::Ptr cloud_filtered (new pcl::PCLPointCloud2 ());
    std::cout << "PointCloud before filtering: " << init_cloud->width * init_cloud->height
       << " data points (" << pcl::getFieldsList (*init_cloud) << "). \n";

    // Create the filtering object
    pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
    sor.setInputCloud(init_cloud);
    sor.setLeafSize(0.01f, 0.01f, 0.01f);
    sor.filter(*cloud_filtered);

    std::cout << "PointCloud after filtering: " << cloud_filtered->width * cloud_filtered->height
       << " data points (" << pcl::getFieldsList (*cloud_filtered) << "). \n";

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
    pfh.setInputCloud(cloud);
    pfh.setInputNormals(cloud_normals);

    // Create an empty kdtree representation, and pass it to the PFH estimation object.
    // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
    // TODO check if you need a new one
    pfh.setSearchMethod(kdtree);

    // Output datasets
    pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs(new pcl::PointCloud<pcl::PFHSignature125>());
    // TODO Better to use for only one point - the center of the object. - computePointPFHSignature

    // Use all neighbors in a sphere of radius 5cm
    // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
    // New by me: make the radius really big to cover the whole object. TODO just take only one point
    pfh.setRadiusSearch (100);

    // Compute the features
    pfh.compute(*pfhs);

    // Find the centroid of the object and all its neighbors within radius search
    // Source: http://pointclouds.org/documentation/tutorials/kdtree_search.php#kdtree-search
//    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree2;
//    Eigen::Vector4f centroid;
//    pcl::compute3DCentroid(*cloud, centroid);
    std::cout << "Point Feature Histograms are ready." << endl;
//    pcl::PointXYZ searchPoint(centroid[0], centroid[1], centroid[2]);

//    Eigen::VectorXf pfh_histogram;
//    std::vector<int> indices(cloud_filtered->width * cloud_filtered->height);
//    for (int i = 0; i < cloud_filtered->width * cloud_filtered->height; i++){
//        indices[i] = i;
//    }
//    pfh.computePointPFHSignature(cloud, cloud_normals, &indices, 5, &pfh_histogram);

    /////////////////////////////////// Visualisation ////////////////////////
    // Visualize the pfh
//    pcl::visualization::PCLHistogramVisualizer hist_visualizer;
//    hist_visualizer.addFeatureHistogram(*pfhs, 125);

    // Visualize the Filtered PointCloud
//    pcl::visualization::CloudViewer viewer ("Simple Cloud Viewer");
//    viewer.showCloud(cloud);
//    while (!viewer.wasStopped ())
//    {
//    }
    // Write the data in a csv file.
    write_file << object_name + "_" + id1 + "_" + id2 + "_" + id3 << ",";
    for(int i = 0; i < 125; i++){
        write_file << pfhs->points[0].histogram[i] << ",";
    }
    // TODO make it with enum, in order to be more elegant.
    if(object_name == "cap")
        write_file << "0\n";
    else if(object_name == "apple")
        write_file << "1\n";
    else if(object_name == "camera")
        write_file << "2\n";
    else if(object_name == "cell_phone")
        write_file << "3\n";
    else if(object_name == "coffee_mug")
        write_file << "4\n";
    else if(object_name == "garlic")
        write_file << "5\n";
    else if(object_name == "bowl")
        write_file << "6\n";
    else if(object_name == "calculator")
        write_file << "7\n";
    return 1;
}


int main (int argc, char** argv)
{
    ofstream csv_file;
    csv_file.open ("Database/pfh.csv");
    if (csv_file.fail()){
        std::cout << "Couldn't open the csv file!";
        return -1;
    }
    csv_file << "Name,";
    for(int i = 1; i < 126; i++){
        csv_file << "Feature" << i << ",";
    }
    csv_file << "Id\n";
    std::ostringstream convert;   // stream used for the conversion

	// Create a initializer list of strings
	// Initialize String Array
    std::string objects[8] = {"apple", "cap", "camera", "cell_phone", "coffee_mug", "garlic", "bowl", "calculator"};
	// Create & Initialize a list with initializer_list object
    for(int object_id = 0; object_id < 8; object_id++){
        for(int i = 1; i < 2; i++){
            for (int j = 1; j < 50; j++){
                convert << i;
                std::string i_string = convert.str();
                convert.str("");
                convert.clear();
                convert << j;
                std::string j_string = convert.str();
                convert.str("");
                convert.clear();
                find_a_name_later(objects[object_id], "1", i_string, j_string, csv_file);
            }
        }
    }
//    find_a_name_later("cap", "1", "1", "110", csv_file);
//    find_a_name_later("apple", "1", "1", "110", csv_file);
    csv_file.close();
    return 0;
}
