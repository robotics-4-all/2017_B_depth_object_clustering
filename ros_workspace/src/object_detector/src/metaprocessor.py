#!/usr/bin/python
import cv2
import numpy as np
import yaml


def inter_section(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w < 0 or h < 0:
        return ()
    return x, y, w, h


def remove_array(l, arr):
    ind = 0
    size = len(l)
    while ind != size and not np.array_equal(l[ind], arr):
        ind += 1
    if ind != size:
        l.pop(ind)
    else:
        raise ValueError('Array not found in list.')
  

def meta_processor(img, rgb_img, img_depth, n_clusters):
    with open("../cfg/conf.yaml", 'r') as stream:
        try:
            doc = yaml.load(stream)
            coldict = doc["clustering"]["coldict"]

            imgproc = rgb_img.copy()
            height, width, channels = img.shape
            overall_mask = np.zeros((height, width), np.uint8)  # blank mask
            object_counter = 0

            prefinal_contours = list()
            for i in range(0, n_clusters):
                desired_color = coldict[i]

                desired_color_array = np.array(desired_color, dtype="uint8")
                mask_init = cv2.inRange(img, desired_color_array, desired_color_array)
                kernel_closing = np.ones((10, 10), np.uint8)
                kernel_opening = np.ones((5, 5), np.uint8)
                kernel_grad = np.ones((5, 5), np.uint8)
                kernel_ero = np.ones((3, 3), np.uint8)

                # Apply morphological operators to get the contour of all labeled areas
                mask = cv2.morphologyEx(mask_init, cv2.MORPH_OPEN, kernel_opening)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_closing)
                mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel_grad)
                mask = cv2.erode(mask, kernel_ero, iterations=1)
                mask = cv2.erode(mask, np.ones((2, 2), np.uint8), iterations=1)

                image, contours, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours[0:len(contours)]:  # TODO fix this, every contour is double
                    x, y, w, h = cv2.boundingRect(c)
                    if w * h > 500:
                        prefinal_contours.append(c)
                overall_mask = np.bitwise_or(mask, overall_mask)

            final_contours = list(prefinal_contours)
            # Check for inter_section of bounding boxes
            for i in range(len(prefinal_contours) - 1):
                box1 = cv2.boundingRect(prefinal_contours[i])
                for j in range(i + 1, len(prefinal_contours)):
                    box2 = cv2.boundingRect(prefinal_contours[j])
                    inter_sec = inter_section(box1, box2)
                    # Check if there is an inter_section and it's larger than the half of the minimum of bounding boxes
                    if inter_sec != () and inter_sec[2] * inter_sec[3] > 0.5 * min((box1[2] * box1[3]),
                                                                                   (box2[2] * box2[3])):
                        center1 = np.asarray(tuple(map(lambda m, n: m + n / 2, box1[0:2], box1[2:24])))
                        center2 = np.asarray(tuple(map(lambda m, n: m + n / 2, box2[0:2], box2[2:24])))
                        dist1 = [center1[0] - inter_sec[0], center1[1] - inter_sec[1]]
                        dist2 = [center2[0] - inter_sec[0], center2[1] - inter_sec[1]]
                        # Vectors that lead from the inter_section point to the center of bounding box
                        unit_vector1 = np.sign(dist1)
                        unit_vector2 = np.sign(dist2)
                        pixel1 = np.asarray(inter_sec[0:2]) + unit_vector1 * 15
                        pixel2 = np.asarray(inter_sec[0:2]) + unit_vector2 * 15

                        # Compare the depths of bounding box in a neighborhood and compare it with a threshold
                        if abs(int(img_depth[pixel1[1]][pixel1[0]]) - int(img_depth[pixel2[1]][pixel2[0]])) < 1:
                            # Find the smallest bounding box, check if it is already removed and remove it.
                            if cv2.contourArea(prefinal_contours[i]) < cv2.contourArea(prefinal_contours[j]) \
                                    and any((np.array_equal(prefinal_contours[i], x)) for x in final_contours):
                                remove_array(final_contours, prefinal_contours[i])
                            elif cv2.contourArea(prefinal_contours[i]) >= cv2.contourArea(prefinal_contours[j]) \
                                    and any((np.array_equal(prefinal_contours[j], x)) for x in final_contours):
                                remove_array(final_contours, prefinal_contours[j])
            for c in final_contours:
                object_counter += 1
                # Get the bounding rect
                x, y, w, h = cv2.boundingRect(c)
                # Draw rectangle to visualize the bounding rect with color
                cv2.rectangle(imgproc, (x, y), (x + w, y + h), coldict[object_counter], 1)
                cv2.putText(imgproc, str(object_counter), (x, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            coldict[object_counter], 1)

            print ("Number of objects detected: " + str(object_counter))
            img_mask = cv2.bitwise_and(rgb_img, rgb_img, mask=cv2.bitwise_not(overall_mask))
            vis1 = np.concatenate((rgb_img, cv2.cvtColor(img_depth, cv2.COLOR_GRAY2BGR), img), axis=1)
            vis2 = np.concatenate((cv2.cvtColor(overall_mask, cv2.COLOR_GRAY2BGR), img_mask, imgproc), axis=1)
            finalvis = np.concatenate((vis1, vis2), axis=0)
            return [finalvis, final_contours]
        except yaml.YAMLError as exc:
            print(exc)
  
if __name__ == '__main__':
    with open("../cfg/conf.yaml", 'r') as Stream:
        try:
            Doc = yaml.load(Stream)
            N_clusters = Doc["clustering"]["number_of_clusters"]
            Rgb_name = Doc["images"]["rgbname"]
            Depth_name = Doc["images"]["depthname"]
            Clustered_name = Doc["images"]["clusteredname"]
            Img = cv2.imread(Clustered_name)
            Rgb_img = cv2.imread(Rgb_name)
            Img_depth = cv2.imread(Depth_name, cv2.IMREAD_GRAYSCALE)

            Vis = meta_processor(Img, Rgb_img, Img_depth, N_clusters)
            cv2.imshow("Image after processing", Vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except yaml.YAMLError as Exc:
            print(Exc)
