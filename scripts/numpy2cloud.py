#!/usr/bin/env python
from data import *

import rospy
import rosbag

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header

image_rows_low = 16 # 8, 16, 32
image_rows_high = 64 # 16, 32, 64
image_rows_full = 64

# ouster
ang_res_x = 360.0/float(image_cols) # horizontal resolution
ang_res_y = 33.2/float(image_rows_high-1) # vertical resolution
ang_start_y = 16.6 # bottom beam angle
sensor_noise = 0.03
max_range = 80.0
min_range = 2.0

upscaling_factor = int(image_rows_high / image_rows_low)

class PointCloudProcessor:
    """Process Point Cloud"""

    def __init__(self):
        self.pubCloud = rospy.Publisher("range_cloud", PointCloud2, queue_size=2)
        self.pubCloud2 = rospy.Publisher("range_cloud_pred", PointCloud2, queue_size=2)
        self.pubCloud3 = rospy.Publisher("range_cloud_pred_reduced_noise", PointCloud2, queue_size=2)
        self.pubCloud4 = rospy.Publisher("range_cloud_small", PointCloud2, queue_size=2)

        self.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        self.rowList = []
        self.colList = []
        for i in range(image_rows_high):
            self.rowList = np.append(self.rowList, np.ones(image_cols)*i)
            self.colList = np.append(self.colList, np.arange(image_cols))

        self.verticalAngle = np.float32(self.rowList * ang_res_y) - ang_start_y
        self.horizonAngle = - np.float32(self.colList + 1 - (image_cols/2)) * ang_res_x + 90.0
        
        self.verticalAngle = self.verticalAngle / 180.0 * np.pi
        self.horizonAngle = self.horizonAngle / 180.0 * np.pi

        self.intensity = self.rowList + self.colList / image_cols
        
    def publishPointCloud(self, thisImage, pubHandle, timeStamp, height):

        if pubHandle.get_num_connections() == 0:
            return

        # multi-channel range image, the first channel is range
        if len(thisImage.shape) == 3:
            thisImage = thisImage[:,:,0]

        lengthList = thisImage.reshape(image_rows_high*image_cols)
        lengthList[lengthList > max_range] = 0.0
        lengthList[lengthList < min_range] = 0.0

        x = np.sin(self.horizonAngle) * np.cos(self.verticalAngle) * lengthList
        y = np.cos(self.horizonAngle) * np.cos(self.verticalAngle) * lengthList
        z = np.sin(self.verticalAngle) * lengthList + height
        
        points = np.column_stack((x,y,z,self.intensity))
        # delete points that has range value 0
        points = np.delete(points, np.where(lengthList==0), axis=0) # comment this line for visualize at the same speed (for video generation)

        header = Header()
        header.frame_id = 'base_link'
        header.stamp = timeStamp

        laserCloudOut = pc2.create_cloud(header, self.fields, points)
        pubHandle.publish(laserCloudOut)

    def test_comparison(self):

        # load images
        # ground truth image: n x rows x cols x 1
        origImages = np.load(os.path.join(home_dir, 'Documents', project_name, 'ouster_range_image.npy')) # 
        # predicted image: n x rows x cols x 2
        predImages = np.load(os.path.join(home_dir, 'Documents', project_name, 'ouster_range_image-from-16-to-64_prediction.npy')) * normalize_ratio # 

        print 'Input range image shape:'
        print origImages.shape
        print predImages.shape

        low_res_index = range(0, image_rows_high, upscaling_factor)
        predImages[:,low_res_index,:,0:1] = origImages[:,low_res_index,:,0:1] # copy some of beams from origImages to predImages
        origImagesSmall = np.zeros(origImages.shape, dtype=np.float32) # for visualizing NN input images
        origImagesSmall[:,low_res_index] = origImages[:,low_res_index]

        predImagesNoiseReduced = np.copy(predImages[:,:,:,0:1])

        # remove noise
        if len(predImages.shape) == 4 and predImages.shape[-1] == 2:
            noiseLabels = predImages[:,:,:,1:2]
            predImagesNoiseReduced[noiseLabels > predImagesNoiseReduced * 0.03] = 0 # after noise removal
            predImagesNoiseReduced[:,low_res_index] = origImages[:,low_res_index]

        predImages[:,:,:,1:2] = None

        for i in range(0, len(origImages), 1):

            timeStamp = rospy.Time.now()

            self.publishPointCloud(origImages[i],               self.pubCloud,  timeStamp, 0)
            self.publishPointCloud(origImagesSmall[i],          self.pubCloud4, timeStamp, 5)
            self.publishPointCloud(predImages[i],               self.pubCloud2, timeStamp, 10)
            self.publishPointCloud(predImagesNoiseReduced[i],   self.pubCloud3, timeStamp, 15)

            if rospy.is_shutdown():
                return

            print ('Displaying {} th of {} images'.format(i, len(origImages)) )
            # raw_input('Displaying {} th of {} images. Press ENTER to continue ... '.format(i, len(origImages)) )
  

if __name__ == '__main__':

    rospy.init_node("cloud_reader")
    
    reader = PointCloudProcessor()

    reader.test_comparison()