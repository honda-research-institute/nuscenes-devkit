import os
import sys
import numpy as np
import cv2
import pickle
from pyquaternion import Quaternion
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'python-sdk'))
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud

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
    mat = np.zeros(3,4)
    mat[:3,:3] = quat2mat3(q)
    mat[:3,3] = t
    return mat

# q is size-4 array for rotation in quaternion, t is size-3 array for translation
# returns 4x4 transform matrix in kitti format
def qt2mat4(q, t):
    mat = quat2mat4(q)
    mat[:3,3] = t
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
        self.P = P # 3x4
        self.P = np.reshape(self.P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = V2C # 3x4
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
        #print("pts_3d_cam=", pts_3d_cam)
        frontal_mask = pts_3d_cam[:, 2] > 0
        if not frontal_mask.all(): # if behind camera, make invalid (negative)
            return -100 * np.ones((pts_3d_velo.shape[0],2))
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

DATAROOT = '/data/sets/nuscenes'
READ_IMG = True
WRITE_PCD = True
type_whitelist = ['Pedestrian', 'Car', 'Truck']
nusc_file = os.path.join(ROOT_DIR, "data", "nusc.pickle")
if os.path.isfile(nusc_file):
    nusc = pickle.load(open(nusc_file, "rb"))
else:
    nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=True)
    pickle.dump(nusc, open(nusc_file, "wb"))

id_list = []  # sample's token
box2d_list = []  # [xmin,ymin,xmax,ymax]
box3d_list = []  # (8,3) array in rect camera coord
input_list = []  # channel number = 4, xyz,intensity in rect camera coord
label_list = []  # 1 for roi object, 0 for clutter
type_list = []  # string e.g. Car
heading_list = []  # ry (along y-axis in rect camera coord)
box3d_size_list = []  # array of l,w,h
frustum_angle_list = []  # angle of 2d box center from pos x-axis
sensor_cam_front = 'CAM_FRONT'
sensor_lidar_top = 'LIDAR_TOP'
for i, my_sample in enumerate(nusc.sample):
    if i<100: continue
    if i>100: break
    #nusc.list_sample(nusc.sample[0]['token'])
    sample_token = my_sample['token']
    #data = my_sample['data']

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
    #print("P=", P)
    #print("V2C=", V2C)
    calib = Calibration(P, V2C)

    ## Transform and filter lidar data
    pc_rect_full = calib.velo2cam(pc_velo[:, :3]) # convert to cam frame
    pc_frontal_mask = pc_rect_full[:, 2] > 0 # Remove pts behind image plane
    pc_velo = pc_velo[pc_frontal_mask, :]  # Nx4
    pc_rect = np.zeros((np.count_nonzero(pc_frontal_mask), 4))
    pc_rect[:, 0:3] = pc_rect_full[pc_frontal_mask, :]
    pc_rect[:, 3] = pc_velo[:, 3]
    _, pc_image_coord, img_fov_inds = calib.get_lidar_in_image_fov( \
        pc_rect[:, 0:3], 0, 0, img_w, img_h)
    if WRITE_PCD:
        write2pcd(pc_rect, "/home/jhuang/pc_rect.pcd")

    ## Get annotations in lidar frame
    _, boxes, _ = nusc.get_sample_data(my_sample['data'][sensor_lidar_top]) # List[Box]
    for box in boxes:
        # reject this box if not in whitelisted categroy
        if box.name not in category2str: continue
        #if category2str[box.name] not in type_whitelist: continue

        box3d_center = box.center
        corners_3d = np.transpose(box.corners()) # 8x3
        #print("corners_3d=", corners_3d)
        corners_2d = calib.project_velo2cam(corners_3d) # 8x2
        #print("corners_2d=", corners_2d)
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
        if box_fov_inds.any():
            print("cat={}, xmin={:d}, ymin={:d}, xmax={:d}, ymax={:d}".format(box.name, xmin, ymin, xmax, ymax))
            #print("center=({:.1f}, {:.1f}, {:.1f})".format(box3d_center[0], box3d_center[1], box3d_center[2]))
            #print("frustum_angle=({:.2f})".format(frustum_angle))
            _, inds = extract_pc_in_box3d(pc_in_box_fov, corners_3d)
            label = np.zeros((pc_in_box_fov.shape[0]))
            label[inds] = 1

            # Get 3D BOX heading
            quat = Quaternion(box.orientation)
            print("orientation={:d} deg".format(int(quat.degrees)))
            # we use same notation as kitti, i.e. rotation around Y-axis in cam coordinates
            # [-pi .. pi], with X-axis being zero.
            heading_angle = -np.deg2rad(quat.degrees)
            # Get 3D BOX size
            box3d_size = box.wlh[[1,0,2]] # l,w,h

