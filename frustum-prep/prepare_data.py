import os
import sys
import numpy as np
import cv2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'python-sdk'))
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud

nusc = NuScenes(version='v1.0-trainval', dataroot='/data/sets/nuscenes', verbose=True)
#nusc.list_sample(nusc.sample[0]['token'])
my_sample = nusc.sample[0]
#data = my_sample['data']

## Get image data
sensor_cam_front = 'CAM_FRONT'
cam_front_data = nusc.get('sample_data', my_sample['data'][sensor_cam_front])
img = cv2.imread(cam_front_data['filename'])

## Get lidar data
sensor_lidar_top = 'LIDAR_TOP'
lidar_top_data = nusc.get('sample_data', my_sample['data'][sensor_lidar_top])
pc = LidarPointCloud.from_file(lidar_top_data['filename'])

## Get annotations
boxes = nusc.get_boxes(cam_front_data['token']) # List[Box]

## Get calib
cam_calib = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
cam_calib['camera_intrinsic']
cam_calib['rotation']
cam_calib['translation']
lidar_calib = nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])
lidar_calib['rotation']
lidar_calib['translation']