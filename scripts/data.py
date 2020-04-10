#!/usr/bin/env python
import os
import time
import thread
import math
import cv2
import numpy as np


data_set = 'carla_ouster_range_image' # high-res training dataset, obtained from simulation - CARLA
test_set = data_set[6:] # low-res testing dataset, obtained from real-world

# Global Variables
image_rows_low = 16 # 8, 16, 32
image_rows_high = 64 # 16, 32, 64
image_cols = 1024
channel_num = 1

upscaling_factor = int(image_rows_high / image_rows_low)
print('#'*50)
print('Input image resolution:   {} by {}'.format(image_rows_low, image_cols))
print('Output image resolution:  {} by {}'.format(image_rows_high, image_cols))
print('Upscaling ratio:          {}'.format(upscaling_factor))


# Sensor settings

# ouster
ang_res_x = 360.0/float(image_cols) # horizontal resolution
ang_res_y = 33.2/float(image_rows_high-1) # vertical resolution
ang_start_y = 16.6 # bottom beam angle
sensor_noise = 0.03
max_range = 80.0
min_range = 2.0

normalize_ratio = 100.0
print('#'*50)
print('Sensor minimum range:     {}'.format(min_range))
print('Sensor maximum range:     {}'.format(max_range))
print('Sensor view angle:        {} to {}'.format(-ang_start_y, -ang_start_y + ang_res_y*(image_rows_high-1)))
print('Range normalize ratio:    {}'.format(normalize_ratio))

# home dir
project_name = 'SuperResolution'
home_dir = os.path.expanduser('~')
root_dir = os.path.join(home_dir, 'Documents', project_name)
# Check Path exists
path_lists = [root_dir]
for folder_name in path_lists:
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

# training & testing data
training_data_file_name = os.path.join(home_dir, 'Documents', project_name, data_set + '.npy')
testing_data_file_name = os.path.join(home_dir, 'Documents', project_name, test_set + '.npy')
print('#'*50)
print('Training data-set:        {} '.format(training_data_file_name))
print('Testing data-set:         {}'.format(testing_data_file_name))


def pre_processing_raw_data(data_set_name):
    # load data
    full_res_data = np.load(data_set_name)
    full_res_data = full_res_data.astype(np.float32, copy=True)

    # if low_res data comes in, expand to high_res data first
    if (full_res_data.shape[1] != image_rows_high):
        full_res_data_temp = np.zeros((full_res_data.shape[0], image_rows_high, full_res_data.shape[2], full_res_data.shape[3]), dtype=np.float32)
        full_res_data_temp[:,range(0, image_rows_high, upscaling_factor)] = full_res_data[:,:]
        full_res_data = full_res_data_temp

    # add gaussian noise for [CARLA] data
    if data_set_name == training_data_file_name:
        print('add noise ...')
        noise = np.random.normal(0, sensor_noise, full_res_data.shape) # mu, sigma, size
        noise[full_res_data == 0] = 0
        full_res_data = full_res_data + noise

    # apply sensor range limit
    full_res_data[full_res_data > max_range] = 0
    full_res_data[full_res_data < min_range] = 0

    # normalize data
    full_res_data = full_res_data / normalize_ratio

    return full_res_data


def get_low_res_from_high_res(high_res_data):
    low_res_index = range(0, image_rows_high, upscaling_factor)
    low_res_data = high_res_data[:,low_res_index]
    return low_res_data

def load_train_data():
    train_data = pre_processing_raw_data(training_data_file_name)
    train_data_input = get_low_res_from_high_res(train_data)
    return train_data_input, train_data

def load_test_data():
    test_data = pre_processing_raw_data(testing_data_file_name)
    test_data_input = get_low_res_from_high_res(test_data)
    return test_data_input, test_data



if __name__=='__main__':
    
    pass