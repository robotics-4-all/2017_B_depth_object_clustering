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
	int num_objects = 12;
    std::string objects[num_objects] = {"apple", "cap", "camera", "cell_phone", "coffee_mug", "garlic",
        "bowl", "calculator", "ball", "banana", "food_can", "food_bag"};
	// Create & Initialize a list with initializer_list object
    for(int object_id = 0; object_id < num_objects; object_id++){
        for(int k = 1; k < 3; k++){
            convert << k;
            std::string k_string = convert.str();
            convert.str("");
            convert.clear();
            for(int i = 1; i < 5; i++){
                convert << i;
                std::string i_string = convert.str();
                convert.str("");
                convert.clear();
                for (int j = 1; j < 151; j++){
                    convert << j;
                    std::string j_string = convert.str();
                    convert.str("");
                    convert.clear();
                    pfh_estimator(objects[object_id], k_string, i_string, j_string, csv_file, object_id);
                }
            }
        }
    }
//    pfh_estimator("cap", "1", "1", "110", csv_file);
//    pfh_estimator("apple", "1", "1", "1", csv_file);
    csv_file.close();
    return 0;
}
