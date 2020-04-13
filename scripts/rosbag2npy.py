#!/usr/bin/env python
from data import *

import rospy
import rosbag

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

# range image size, depends on your sensor, i.e., VLP-16: 16x1800, OS1-64: 64x1024
image_rows_full = 64
image_cols = 1024

# Ouster OS1-64 (gen1)
ang_res_x = 360.0/float(image_cols) # horizontal resolution
ang_res_y = 33.2/float(image_rows_full-1) # vertical resolution
ang_start_y = 16.6 # bottom beam angle
max_range = 80.0
min_range = 2.0

# Convert ros bags to npy
def bag2np(data_set_name, pointcloud_topic, npy_file_name):
    """
    convert all .bag files in a specific folder to a single .npy range image file.
    """
    print('#'*50)
    print('Dataset name: {}'.format(data_set_name))
    range_image_array = np.empty([0, image_rows_full, image_cols, 1], dtype=np.float32)
    # find all bag files in the given dir
    bag_file_path = os.path.join(data_set_name)
    bag_files = os.listdir(bag_file_path)
    print bag_files
    # loop through all bags in the given directory
    for file_name in bag_files:
        # bag file path
        file_path = os.path.join(bag_file_path, file_name)
        # open ros bags 
        with rosbag.Bag(file_path, 'r') as bag:
            # loop through all msgs in the bag
            for topic, msg, t in bag.read_messages(topics=[pointcloud_topic]):
                # match point cloud topic
                if topic == pointcloud_topic:
                    print('processing {}th point cloud message...\r'.format(range_image_array.shape[0])),
                    # convert point cloud message to numpy array
                    points_obj = pc2.read_points(msg, skip_nans=True, field_names=("x", "y", "z"))
                    points_array = np.array(list(points_obj), dtype=np.float32)
                    # project points to range image
                    range_image = np.zeros((1, image_rows_full, image_cols, 1), dtype=np.float32)
                    x = points_array[:,0]
                    y = points_array[:,1]
                    z = points_array[:,2]
                    # find row id
                    vertical_angle = np.arctan2(z, np.sqrt(x * x + y * y)) * 180.0 / np.pi
                    relative_vertical_angle = vertical_angle + ang_start_y
                    rowId = np.int_(np.round_(relative_vertical_angle / ang_res_y))
                    # find column id
                    horitontal_angle = np.arctan2(x, y) * 180.0 / np.pi
                    colId = -np.int_((horitontal_angle-90.0)/ang_res_x) + image_cols/2;
                    shift_ids = np.where(colId>=image_cols)
                    colId[shift_ids] = colId[shift_ids] - image_cols
                    # filter range
                    thisRange = np.sqrt(x * x + y * y + z * z)
                    thisRange[thisRange > max_range] = 0
                    thisRange[thisRange < min_range] = 0
                    # save range info to range image
                    for i in range(len(thisRange)):
                        if rowId[i] < 0 or rowId[i] >= image_rows_full or colId[i] < 0 or colId[i] >= image_cols:
                            continue
                        range_image[0, rowId[i], colId[i], 0] = thisRange[i]
                    # append range image to array
                    range_image_array = np.append(range_image_array, range_image, axis=0)

    # save full resolution image array
    np.save(npy_file_name, range_image_array)
    print('Dataset saved: {}'.format(npy_file_name))




if __name__=='__main__':
    
    # put all the bags that can be trained in this folder, the package will combine them as a single npy file
    data_set_name = os.path.join(home_dir, 'Documents', 'bags')
    # your point cloud ros topic in these bag files, i.e., velodyne_points
    pointcloud_topic = '/os1_node/points'
    # the location where your npy file will be saved
    npy_file_name = os.path.join(home_dir, 'Documents', "data.npy")

    bag2np(data_set_name, pointcloud_topic, npy_file_name)