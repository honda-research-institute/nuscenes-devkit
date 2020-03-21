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
print("# train scenes = {}, # val scenes = {}".format(len(scene_splits['train']), len(scene_splits['val'])))

category2str = {'human.pedestrian.adult': 'Pedestrian', \
                'vehicle.car': 'Car', 'vehicle.truck': 'Truck'}


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

DATAROOT = '/data/sets/nuscenes'
READ_IMG = False
WRITE_PCD = False
WRITE_FRUSTUM_PCD = False
PLOT_HISTOGRAM = False
WRITE_PICKLE = True
VERBOSITY = 0
MIN_OBJ_PIX_HEIGHT = 20
MIN_FRUSTUM_PT_COUNT = 5
MAX_DIST = 25  # meters, only use objects up to this distance
type_whitelist = category2str.keys()
nusc_file = os.path.join(ROOT_DIR, "data", "nusc.pickle")
output_filename = os.path.join(ROOT_DIR, "data", "nusc_carpedtruck.pickle")
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

id_list_train, id_list_val = [], []  # sample's token
box2d_list_train, box2d_list_val = [], []  # [xmin,ymin,xmax,ymax]
box3d_list_train, box3d_list_val = [], []  # (8,3) array in rect camera coord
input_list_train, input_list_val = [], []  # channel number = 4, xyz,intensity in rect camera coord
label_list_train, label_list_val = [], []  # 1 for roi object, 0 for clutter
type_list_train, type_list_val = [], []  # string e.g. Car
heading_list_train, heading_list_val = [], []  # ry (along y-axis in rect camera coord)
box3d_size_list_train, box3d_size_list_val = [], []  # array of l,w,h
frustum_angle_list_train, frustum_angle_list_val = [], []  # angle of 2d box center from pos x-axis

sensor_cam_front = 'CAM_FRONT'
sensor_lidar_top = 'LIDAR_TOP'
pt_cnt_stats = {'Pedestrian': [], 'Car': [], 'Truck': []}
pos_cnt = 0
all_cnt = 0
pick = 88
for i, my_sample in enumerate(tqdm.tqdm(nusc.sample)):
    is_train_sample = (my_sample['scene_token'] in train_scene_tokens)

    sample_token = my_sample['token']
    # data = my_sample['data']

    ## Get raw image data
    cam_front_data = nusc.get('sample_data', my_sample['data'][sensor_cam_front])
    img_h = cam_front_data['height']
    img_w = cam_front_data['width']
    if READ_IMG:
        img = cv2.imread(os.path.join(DATAROOT, cam_front_data['filename']))
        cv2.imwrite("/home/jhuang/img.png", img)

    ## Get raw lidar data
    lidar_top_data = nusc.get('sample_data', my_sample['data'][sensor_lidar_top])
    pc = LidarPointCloud.from_file(os.path.join(DATAROOT, lidar_top_data['filename']))
    pc_velo = np.transpose(pc.points)  # Nx4
    if WRITE_PCD:
        write2pcd(pc_velo, "/home/jhuang/pc_velo.pcd")

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
    # print("P=", P)
    # print("V2C=", V2C)
    calib = Calibration(P, V2C)

    ## Transform and filter lidar data
    pc_rect_full = calib.velo2cam(pc_velo[:, :3])  # convert to cam frame
    pc_frontal_mask = pc_rect_full[:, 2] > 0  # Remove pts behind image plane
    pc_velo = pc_velo[pc_frontal_mask, :]  # Nx4
    pc_rect = np.zeros((np.count_nonzero(pc_frontal_mask), 4))
    pc_rect[:, 0:3] = pc_rect_full[pc_frontal_mask, :]
    pc_rect[:, 3] = pc_velo[:, 3]
    _, pc_image_coord, img_fov_inds = calib.get_lidar_in_image_fov( \
        pc_rect[:, 0:3], 0, 0, img_w, img_h)
    if WRITE_PCD:
        write2pcd(pc_rect, "/home/jhuang/pc_rect.pcd")

    ## Get annotations in lidar frame
    _, boxes, _ = nusc.get_sample_data(my_sample['data'][sensor_lidar_top])  # List[Box] in sensor frame
    j = 0
    for box in boxes:
        # reject this box if not in whitelisted category
        if box.name not in category2str: continue
        obj_category = category2str[box.name]

        box3d_center = box.center
        if np.hypot(box3d_center[0], box3d_center[1]) > MAX_DIST:
            continue
        corners_3d = np.transpose(box.corners())  # 8x3
        corners_2d = calib.project_velo2cam(corners_3d)  # 8x2
        corners_x, corners_y = corners_2d[:, 0], corners_2d[:, 1]
        xmin, xmax = int(np.amin(corners_x)), int(np.amax(corners_x))
        ymin, ymax = int(np.amin(corners_y)), int(np.amax(corners_y))
        box_fov_inds = (pc_image_coord[:, 0] < xmax) & \
                       (pc_image_coord[:, 0] >= xmin) & \
                       (pc_image_coord[:, 1] < ymax) & \
                       (pc_image_coord[:, 1] >= ymin)
        box_fov_inds = box_fov_inds & img_fov_inds
        pc_in_box_fov = pc_rect[box_fov_inds, :]
        # Get frustum angle (according to center pixel in 2D BOX)
        box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
        uvdepth = np.zeros((1, 3))
        uvdepth[0, 0:2] = box2d_center
        uvdepth[0, 2] = 20  # some random depth
        box2d_center_rect = calib.project_image_to_rect(uvdepth)
        # FIXME: why the -1 in frustum_angle
        # According to this, right is zero, and clockwise is positive
        frustum_angle = -1 * np.arctan2(box2d_center_rect[0, 2],
                                        box2d_center_rect[0, 0])
        if pc_in_box_fov.shape[0]:
            # print("cat={}, xmin={:d}, ymin={:d}, xmax={:d}, ymax={:d}".format(box.name, xmin, ymin, xmax, ymax))
            # print("center=({:.1f}, {:.1f}, {:.1f})".format(box3d_center[0], box3d_center[1], box3d_center[2]))
            # print("frustum_angle=({:.2f})".format(frustum_angle))
            # print("corners_3d=", corners_3d)
            # print("corners_2d=", corners_2d)
            corners_3d_rect = calib.velo2cam(corners_3d)
            _, inds = extract_pc_in_box3d(pc_in_box_fov, corners_3d_rect)
            label = np.zeros((pc_in_box_fov.shape[0]))
            label[inds] = 1

            # Get 3D BOX heading
            quat = Quaternion(box.orientation)
            if VERBOSITY > 0:
                print("cat={}, x_c={:d}, y_c={:d}, orientation={:d} deg, n_pts={:d}".format( \
                    box.name, int(box2d_center[0]), int(box2d_center[1]), int(quat.degrees), pc_in_box_fov.shape[0]))
            # we use same notation as kitti, i.e. rotation around Y-axis in cam coordinates
            # [-pi .. pi], with X-axis being zero.
            heading_angle = -np.deg2rad(quat.degrees)
            # Get 3D BOX size
            box3d_size = box.wlh[[1, 0, 2]]  # l,w,h

            # Reject too far away object or object with too few points
            if ymax - ymin < MIN_OBJ_PIX_HEIGHT or np.sum(label) < MIN_FRUSTUM_PT_COUNT:
                continue

            if WRITE_FRUSTUM_PCD:
                write2pcd(pc_in_box_fov, "/home/jhuang/pc_frustum.pcd".replace('.pcd', '_{:d}.pcd'.format(j)))
                j += 1

            if is_train_sample:
                id_list_train.append(sample_token)
                box2d_list_train.append(np.array([xmin, ymin, xmax, ymax]))
                box3d_list_train.append(corners_3d_rect)
                input_list_train.append(pc_in_box_fov)
                label_list_train.append(label)
                type_list_train.append(obj_category)
                heading_list_train.append(heading_angle)
                box3d_size_list_train.append(box3d_size)
                frustum_angle_list_train.append(frustum_angle)
            else:
                id_list_val.append(sample_token)
                box2d_list_val.append(np.array([xmin, ymin, xmax, ymax]))
                box3d_list_val.append(corners_3d_rect)
                input_list_val.append(pc_in_box_fov)
                label_list_val.append(label)
                type_list_val.append(obj_category)
                heading_list_val.append(heading_angle)
                box3d_size_list_val.append(box3d_size)
                frustum_angle_list_val.append(frustum_angle)

            # collect statistics
            pos_cnt += np.sum(label)
            all_cnt += pc_in_box_fov.shape[0]
            pt_cnt_stats[obj_category].append(pc_in_box_fov.shape[0])

print('Average pos ratio: %f' % (pos_cnt / float(all_cnt)))
print('Average npoints: %f' % (float(all_cnt) / (len(id_list_train) + len(id_list_val))))

if PLOT_HISTOGRAM:
    # the histogram of the data
    fig, axes = plt.subplots(1, 3)
    n_ped, bins_ped, _ = axes[0].hist(pt_cnt_stats['Pedestrian'], 30, density=True, facecolor='g', alpha=0.75)
    axes[0].set_title('Pedestrian')
    n_car, bins_car, _ = axes[1].hist(pt_cnt_stats['Car'], 30, density=True, facecolor='k', alpha=0.75)
    axes[1].set_title('Car')
    n_truck, bins_truck, _ = axes[2].hist(pt_cnt_stats['Truck'], 30, density=True, facecolor='b', alpha=0.75)
    axes[2].set_title('Truck')
    plt.show()

if WRITE_PICKLE:
    of = output_filename.replace(".pickle", "_train.pickle")
    with open(of, 'wb') as fp:
        dump_pickle2(id_list_train, fp)
        dump_pickle2(box2d_list_train, fp)
        dump_pickle2(box3d_list_train, fp)
        dump_pickle2(input_list_train, fp)
        dump_pickle2(label_list_train, fp)
        dump_pickle2(type_list_train, fp)
        dump_pickle2(heading_list_train, fp)
        dump_pickle2(box3d_size_list_train, fp)
        dump_pickle2(frustum_angle_list_train, fp)
    of = output_filename.replace(".pickle", "_val.pickle")
    with open(of, 'wb') as fp:
        dump_pickle2(id_list_val, fp)
        dump_pickle2(box2d_list_val, fp)
        dump_pickle2(box3d_list_val, fp)
        dump_pickle2(input_list_val, fp)
        dump_pickle2(label_list_val, fp)
        dump_pickle2(type_list_val, fp)
        dump_pickle2(heading_list_val, fp)
        dump_pickle2(box3d_size_list_val, fp)
        dump_pickle2(frustum_angle_list_val, fp)