# Lidar Super-resolution

This repository contains code for lidar super-resolution with ground vehicles driving on roadways, which relies on a driving simulator to enhance the apparent resolution of a physical lidar. To increase the resolution of the point cloud captured by a sparse 3D lidar, we convert this problem from 3D Euclidean space into an image super-resolution problem in 2D image space, which is solved using a deep convolutional neural network. By projecting a point cloud onto a range image, we can efficiently enhance the resolution of such an image using a deep neural network. We train the network purely using computer-generated data (i.e., CARLA simulator). [A video of the package can be found here](https://youtu.be/rNVTpkz2ggY).

# Dependency

The package depends on Numpy, Keras, Tensorflow, and ROS. ROS only is used for visualization.

# Compile

Download the package to your workspace and compile to code with ```catkin_make``` (for visualization only).

# Data Download

[Download the demo data](https://drive.google.com/drive/folders/1nqxnJI1d_u5II4uzXoFJxVMdmbpdHdUh?usp=sharing) into your ```Documents``` folder in your home directory. The data directory should look like this ```/home/$USER/Documents/SuperResolution```. The ```SuperResolution``` is the project directory and contains .npy files. You can also change the directory settings in ```data.py```. We simulate an Ouster OS1-64 lidar in the CARLA simulator. The simulated point clouds are projected onto range images and used for training. We test the trained neulral network on real Ouster data to see its performance.

Demo data:
```
carla_ouster_range_image.npy # simulated dataset for network training (Simulated Ouster OS1-64 lidar in CARLA, train 16-beam to 64-beam)
ouster_range_image.npy # real-world dataset for testing. For example, you have OS1-16 data and want to increase its resolution to 64-beam
ouster_range_image-from-16-to-64_prediction.npy # predicted high-res data using the network trained on the simulated dataset
weights.h5 # an example weight file for the Ouster 16 to 64 neural network
```

# Train Neural Network

Run ```run.py``` to start training the neural network using the provided data or your data.
```
python run.py
```

# Training Tips

You may want to train a network and then perform inference on your own data after you can run the sample data on your computer. To achieve the best performance, the simulated high-resolution lidar in the simulator should be mounted similarly to the mounting of your low-resolution lidar in the real world. For example, if your real lidar is mounted horizontally at 1.5-meter height. The simulated lidar should be placed like this in the simulator. If your real lidar is mounted horizontally, but your simulated lidar is mounted vertically, you will not get good performance as the view of the two lidars are completely different. Also, the field of view of the real lidar and the simulated lidar should be the same. 

Data augmentation improves the performance of the network greatly. When you collect some data by simulating a high-resolution lidar, you can perform some operations like scaling the range values or flipping the range image.

# Prepare Own Data

There is a script provided in this package, ```rosbag2npy.py```, which converts point cloud messages in a rosbag into range images. The detailed usage of the script can be found in the comments of it.

# Visualization

Run the code following to visualize your predictions:
```
roslaunch lidar_super_resolution visualize.launch
```

<p align='center'>
    <img src="/docs/demo.gif" alt="drawing" width="800"/>
</p>

## Cite 

Thank you for citing [our paper](./docs/paper.pdf) if you use any of this code: 
```
@inproceedings{superresolution2020shan,
  title={Simulation-based Lidar Super-resolution for Ground Vehicles},
  author={Shan, Tixiao and Wang, Jinkun and Chen, Fanfei and Szenher, Paul and Englot, Brendan},
  journal={arXiv preprint arXiv:2004.05242}
  year={2020}
}
