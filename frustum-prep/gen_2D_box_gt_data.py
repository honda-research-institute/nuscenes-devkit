import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import cv2
import pickle
import tqdm  # for progress bar
from pyquaternion import Quaternion
from numpy import linalg as LA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'python-sdk'))
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud
from nuscenes.utils import splits

scene_splits = splits.create_splits_scenes()
print("# train scenes = {}, # val scenes = {}".format(len(scene_splits['train']), len(scene_splits['val'])))
category2str = {'vehicle.car': 'Car'}
category2str = {'vehicle.car': 'Car', 'vehicle.truck': 'Truck'}


def in_hull(p, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return pc[box3d_roi_inds, :], box3d_roi_inds


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

    # rotation only
    def velo2cam_rot(self, pts_3d):
        return np.dot(pts_3d, np.transpose(self.V2C[:3, :3]))

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

    def is_velo_inside_cam_fov(self, pts_3d_velo, xmin, ymin, xmax, ymax):
        ''' Input: pt_3d in velo coord.
            Output: True/False
        '''
        pt_3d_cam = self.velo2cam(pts_3d_velo[None, :])
        if pt_3d_cam[0, 2] <= 0:
            return False
        pt_2d = self.cam3dto2d(pt_3d_cam)
        return (pt_2d[0,0] < xmax) and (pt_2d[0,0] >= xmin) and \
               (pt_2d[0,1] < ymax) and (pt_2d[0,1] >= ymin)

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


def hist_percent(data, title, ax = None):
    if not ax:
        ax = plt.gca()

    ax.hist(data, 20, weights=np.ones(len(data)) / len(data))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title(title)


DATAROOT = '/data/sets/nuscenes'
type_whitelist = category2str.keys()
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
sensor_lidar_top = 'LIDAR_TOP'
ann_map_train, ann_map_val = dict(), dict()
for i, my_sample in enumerate(tqdm.tqdm(nusc.sample)):
    sample_token = my_sample['token']
    is_train_sample = (my_sample['scene_token'] in train_scene_tokens)

    ## Get raw image data
    cam_front_data = nusc.get('sample_data', my_sample['data'][sensor_cam_front])
    img_name = os.path.basename(cam_front_data['filename'])
    img_name, _ = os.path.splitext(img_name) # strip extension
    img_h = cam_front_data['height']
    img_w = cam_front_data['width']
    img = cv2.imread(os.path.join(DATAROOT, cam_front_data['filename']))

    ## Get raw lidar data
    lidar_top_data = nusc.get('sample_data', my_sample['data'][sensor_lidar_top])

    ## Get calib
    cam_calib = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
    intrinsic = cam_calib['camera_intrinsic']  # 3x3 list
    intrinsic = np.reshape(intrinsic, (3, 3))
    P = np.zeros((3, 4))
    P[:3, :3] = intrinsic  # append a column of zeros to instrinsic matrix
    rot = cam_calib['rotation']  # 4 list
    trans = cam_calib['translation']  # 3 list
    T_base_cam = qt2mat4(rot, trans)
    lidar_calib = nusc.get('calibrated_sensor', lidar_top_data['calibrated_sensor_token'])
    rot = lidar_calib['rotation']  # 4 list
    trans = lidar_calib['translation']  # 3 list
    T_base_lidar = qt2mat4(rot, trans)
    V2C = np.matmul(invMat4(T_base_cam), T_base_lidar)[:3, :]
    calib = Calibration(P, V2C)

    ## Get annotations in lidar frame
    _, boxes, _ = nusc.get_sample_data(my_sample['data'][sensor_lidar_top])  # List[Box] in sensor frame
    boxes_list = []
    for box in boxes:
        # reject this box if not in whitelisted category
        if box.name not in category2str:
            continue
        obj_category = category2str[box.name]

        box3d_center = box.center
        # Projection of obj center not in cam FOV
        if not calib.is_velo_inside_cam_fov(box3d_center, 0, 0, img_w, img_h):
            continue

        corners_3d = np.transpose(box.corners())  # 8x3
        corners_2d = calib.project_velo2cam(corners_3d)  # 8x2
        corners_x, corners_y = corners_2d[:, 0], corners_2d[:, 1]
        xmin, xmax = int(np.amin(corners_x)), int(np.amax(corners_x))
        ymin, ymax = int(np.amin(corners_y)), int(np.amax(corners_y))

        if xmin < 0 or xmax >= img_w or ymin < 0 or ymax >= img_h:
            continue

        height_2d = ymax - ymin
        if height_2d < 20:
            continue


        quat = Quaternion(box.orientation)
        yaw = np.deg2rad(quat.degrees)
        heading_vec = np.array([np.cos(yaw), np.sin(yaw), 0]) # in velo frame
        heading_3d_cam = calib.velo2cam_rot(heading_vec[None, :])[0,:]
        center_3d_cam = calib.velo2cam(box3d_center[None, :])[0,:]
        a = heading_3d_cam[[0,2]]
        b = center_3d_cam[[0,2]]
        cos_val = abs(a.dot(b) / (LA.norm(a) * LA.norm(b)))
        if cos_val > 0.86:
            heading_cls = 0
        elif cos_val < 0.5:
            heading_cls = 2
        else:
            heading_cls = 1

        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        print("x={:d}, y={:d}, orientation={:d} deg".format( \
            int(box2d_center[0]), int(box2d_center[1]), int(quat.degrees)))
        print("a={:.1f} {:.1f}, b={:.1f} {:.1f}".format(a[0], a[1], b[0], b[1]))
        boxes_list.append([xmin, ymin, xmax, ymax, obj_category, heading_cls])

    plt.imshow(img)
    plt.show()
    if is_train_sample:
        ann_map_train[img_name] = boxes_list
    else:
        ann_map_val[img_name] = boxes_list

train_output_filename = os.path.join(ROOT_DIR, "data", "nusc_car_boxes_train.pickle")
val_output_filename = os.path.join(ROOT_DIR, "data", "nusc_car_boxes_val.pickle")
with open(train_output_filename, 'wb') as fp:
    pickle.dump(ann_map_train, fp)
with open(val_output_filename, 'wb') as fp:
    pickle.dump(ann_map_val, fp)