import os
import sys
import numpy as np
import pickle
import tqdm  # for progress bar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'python-sdk'))
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils import splits
scene_splits = splits.create_splits_scenes()
print("# train scenes = {}".format(len(scene_splits['train'])))
print("# val scenes = {}".format(len(scene_splits['val'])))

DATAROOT = '/data/sets/nuscenes'
nusc_file = os.path.join(ROOT_DIR, "data", "nusc.pickle")
if os.path.isfile(nusc_file):
    print("Loading {} ...".format(nusc_file))
    nusc = pickle.load(open(nusc_file, "rb"))
else:
    print("Loading from {} ...".format(DATAROOT))
    nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=True)
    pickle.dump(nusc, open(nusc_file, "wb"))
print("Done")

# Convert scene names to tokens
train_scene_tokens, val_scene_tokens = set(), set()
for scene in nusc.scene:
    if scene['name'] in scene_splits['train']:
        train_scene_tokens.add(scene['token'])
    else:
        assert scene['name'] in scene_splits['val']
        val_scene_tokens.add(scene['token'])

sensor_cam_front = 'CAM_FRONT'
f_out_train = open(os.path.join(DATAROOT,'train.lst'),'w')
f_out_val = open(os.path.join(DATAROOT,'val.lst'),'w')
for i, my_sample in enumerate(tqdm.tqdm(nusc.sample)):
    sample_token = my_sample['token']
    cam_front_data = nusc.get('sample_data', my_sample['data'][sensor_cam_front])
    img_name = os.path.basename(cam_front_data['filename'])
    img_name, _ = os.path.splitext(img_name) # strip extension

    sc_token = my_sample['scene_token']
    if (sc_token in train_scene_tokens):
        f_out_train.write("{}\n".format(img_name))
    else:
        assert sc_token in val_scene_tokens
        f_out_val.write("{}\n".format(img_name))

f_out_train.close()
f_out_val.close()