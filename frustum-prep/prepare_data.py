import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'python-sdk'))
from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-trainval', dataroot='/data/sets/nuscenes', verbose=True)
#nusc.list_sample(nusc.sample[0]['token'])
my_sample = nusc.sample[0]
#data = my_sample['data']
sensor = 'CAM_FRONT'LIDAR_TOP
cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
sensor = 'LIDAR_TOP'
lidar_top_data = nusc.get('sample_data', my_sample['data'][sensor])
anns = my_sample['anns']