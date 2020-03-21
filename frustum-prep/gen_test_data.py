import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import tqdm  # for progress bar
from pyquaternion import Quaternion

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

# from Kitti
def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr


def invMat4(T44):
    T34 = T44[:3, :]
    T34_inv = inverse_rigid_trans(T34)
    T44_inv = np.identity(4)
    T44_inv[:3, :] = T34_inv
    return T44_inv


# q is size-4 array, returns 3x3 rot matrix
def quat2mat3(q):
    quat = Quaternion(q)
    return quat.rotation_matrix


# q is size-4 array, returns 4x4 T-matrix
def quat2mat4(q):
    quat = Quaternion(q)
    return quat.transformation_matrix


# q is size-4 array for rotation in quaternion, t is size-3 array for translation
# returns 3x4 transform matrix in kitti format
def qt2mat34(q, t):
    mat = np.zeros(3, 4)
    mat[:3, :3] = quat2mat3(q)
    mat[:3, 3] = t
    return mat


# q is size-4 array for rotation in quaternion, t is size-3 array for translation
# returns 4x4 transform matrix in kitti format
def qt2mat4(q, t):
    mat = quat2mat4(q)
    mat[:3, 3] = t
    return mat


# scan is Nx4 array
def write2pcd(scan, pcd_file):
    n_pts = scan.shape[0]
    with open(pcd_file, "w") as f:
        f.write('# .PCD v0.7 - Point Cloud Data file format\n')
        f.write('VERSION 0.7\n')
        f.write('FIELDS x y z x_origin y_origin z_origin\n')
        f.write('SIZE 4 4 4 4 4 4\n')
        f.write('TYPE F F F F F F\n')
        f.write('COUNT 1 1 1 1 1 1\n')
        f.write('WIDTH ' + str(n_pts) + "\n")
        f.write('HEIGHT 1\n')
        f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        f.write('POINTS ' + str(n_pts) + "\n")
        f.write('DATA ascii\n')
        for i in range(n_pts):
            f.write("{} {} {} 0 0 0\n".format(scan[i, 0], scan[i, 1], scan[i, 2]))


class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.

        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref

        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]

        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)

        velodyne coord:
        front x, left y, up z

        rect/ref camera coord:
        right x, down y, front z

        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf

        TODO(rqi): do matrix multiplication only once for each projection.
    '''

    # P is 3x4 camera instrinsic matrix
    # V2C is 3x4 extrinsic matrix to convert P_velo to P_cam
    def __init__(self, P, V2C):
        # Projection matrix from rect camera coord to image2 coord
        self.P = P  # 3x4
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = V2C  # 3x4
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = np.eye(3)

        # Camera intrinsics and extrinsics
        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)  # relative
        self.b_y = self.P[1, 3] / (-self.f_v)

    # Convert pts from velo frame to cam frame
    # input/output: both Nx3
    def velo2cam(self, pts_3d):
        pts_4d = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))  # nx4
        return np.dot(pts_4d, np.transpose(self.V2C))

    # ===========================
    # ------- 3d to 2d ----------
    # ===========================
    def cam3dto2d(self, pts_3d):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_4d = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))  # nx4
        pts_2d_hom = np.dot(pts_4d, np.transpose(self.P))  # nx3 homogeneous form
        pts_2d_hom[:, 0] /= pts_2d_hom[:, 2]
        pts_2d_hom[:, 1] /= pts_2d_hom[:, 2]
        return pts_2d_hom[:, 0:2]

    def project_velo2cam(self, pts_3d_velo):
        ''' Input: nx3 points in velo coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_cam = self.velo2cam(pts_3d_velo)
        # print("pts_3d_cam=", pts_3d_cam)
        frontal_mask = pts_3d_cam[:, 2] > 0
        if not frontal_mask.all():  # if behind camera, make invalid (negative)
            return -100 * np.ones((pts_3d_velo.shape[0], 2))
        else:
            return self.cam3dto2d(pts_3d_cam)

    # pts_3d: 3D points in cam coordinate
    def get_lidar_in_image_fov(self, pts_3d, xmin, ymin, xmax, ymax):
        ''' Filter lidar points, keep those in image FOV '''
        pts_2d = self.cam3dto2d(pts_3d)
        fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
                   (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
        imgfov_pc_velo = pts_3d[fov_inds, :]
        return imgfov_pc_velo, pts_2d, fov_inds

    # ===========================
    # ------- 2d to 3d ----------
    # ===========================
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - self.c_u) * uv_depth[:, 2]) / self.f_u + self.b_x
        y = ((uv_depth[:, 1] - self.c_v) * uv_depth[:, 2]) / self.f_v + self.b_y
        pts_3d_rect = np.zeros((n, 3))
        pts_3d_rect[:, 0] = x
        pts_3d_rect[:, 1] = y
        pts_3d_rect[:, 2] = uv_depth[:, 2]
        return pts_3d_rect


# Frustum-Pnet requires old pickle protocol
def dump_pickle2(data, fp):
    pickle.dump(data, fp, protocol=2)

DATAROOT = '/data/sets/nuscenes/test'
WRITE_IMG = True
WRITE_PCD = True
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
    if WRITE_IMG:
        img = cv2.imread(os.path.join(DATAROOT, cam_front_data['filename']))
        cv2.imwrite(os.path.join(SAVEDIR, "{:05d}.png".format(j)), img)

    ## Get raw lidar data
    lidar_top_data = nusc.get('sample_data', my_sample['data'][sensor_lidar_top])
    pc = LidarPointCloud.from_file(os.path.join(DATAROOT, lidar_top_data['filename']))
    pc_velo = np.transpose(pc.points)  # Nx4
    if WRITE_PCD:
        write2pcd(pc_velo, os.path.join(SAVEDIR, "{:05d}.pcd".format(j)))