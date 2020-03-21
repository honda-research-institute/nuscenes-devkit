import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import tqdm  # for progress bar
from shutil import copyfile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'python-sdk'))
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils import splits
scene_splits = splits.create_splits_scenes()
print("# test scenes = {}".format(len(scene_splits['test'])))

category2str = {'human.pedestrian.adult': 'Pedestrian', \
                'vehicle.car': 'Car', 'vehicle.truck': 'Truck'}


DATAROOT = '/data/sets/nuscenes/test'
SAVEDIR = os.path.join(DATAROOT, 'extract')
type_whitelist = category2str.keys()
print("Loading from {} ...".format(DATAROOT))
nusc = NuScenes(version='v1.0-test', dataroot=DATAROOT, verbose=True)
print("Done")

# Convert scene names to tokens
test_scene_tokens = set()
for scene in nusc.scene:
    if scene['name'] in scene_splits['test']:
        test_scene_tokens.add(scene['token'])

sensor_cam_front = 'CAM_FRONT'
sensor_lidar_top = 'LIDAR_TOP'
pt_cnt_stats = {'Pedestrian': [], 'Car': [], 'Truck': []}
cnt = -1
for i, my_sample in enumerate(tqdm.tqdm(nusc.sample)):
    if (my_sample['scene_token'] not in test_scene_tokens):
        continue

    cnt += 1
    if cnt % 100 != 0:
        continue
    else:
        j = int(cnt / 100)

    sample_token = my_sample['token']

    ## Get raw image data
    cam_front_data = nusc.get('sample_data', my_sample['data'][sensor_cam_front])
    img_h = cam_front_data['height']
    img_w = cam_front_data['width']
    img = cv2.imread(os.path.join(DATAROOT, cam_front_data['filename']))
    cv2.imwrite(os.path.join(SAVEDIR, "{:05d}.png".format(j)), img)

    ## Get raw lidar data
    lidar_top_data = nusc.get('sample_data', my_sample['data'][sensor_lidar_top])
    src = os.path.join(DATAROOT, lidar_top_data['filename'])
    dst = os.path.join(SAVEDIR, "{:05d}.bin".format(j))
    copyfile(src, dst)