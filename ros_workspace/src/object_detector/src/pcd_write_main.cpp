#include "../include/object_detector/pcd_write.hpp"

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
	int num_objects = 17;
//    std::string objects[num_objects] = {"apple", "ball", "banana", "bell_pepper", "binder", "bowl", "calculator",
//        "camera", "cap", "cell_phone", "cereal_box", "coffee_mug", "comb", "dry_battery", "food_can", "food_bag",
//        "garlic"};
    std::string objects[num_objects] = {"apple", "banana", "binder", "calculator",
        "camera", "food_can", "food_bag"};
	// Create & Initialize a list with initializer_list object
    for(int object_id = 0; object_id < num_objects; object_id++){
        for(int k = 1; k < 6; k++){
            convert << k;
            std::string k_string = convert.str();
            convert.str("");
            convert.clear();
            for(int i = 1; i < 3; i++){
                convert << i;
                std::string i_string = convert.str();
                convert.str("");
                convert.clear();
                for (int j = 1; j < 51; j++){
                    convert << j;
                    std::string j_string = convert.str();
                    convert.str("");
                    convert.clear();
                    pfh_estimator(objects[object_id], k_string, i_string, j_string, csv_file, object_id);
                }
            }
        }
    }
    csv_file.close();
    return 0;
}
