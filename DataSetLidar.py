# %%
# -------------------------------------------------------------------
# Copyright (C) 2020 Università degli studi di Milano-Bicocca, iralab
# Author: Daniele Cattaneo (d.cattaneo10@campus.unimib.it)
# Released under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# http://creativecommons.org/licenses/by-nc-sa/4.0/
# -------------------------------------------------------------------

# Modified Author: Xudong Lv
# based on github.com/cattaneod/CMRNet/blob/master/DatasetVisibilityKitti.py

# Modified Author: Johannes Berner
# based on github.com/LvXudong-HIT/LCCNet/blob/main/DatasetLidarCamera.py

import csv
import os
from math import radians
import cv2

import mathutils
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TTF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

#from utils import invert_pose, rotate_forward, quaternion_from_matrix  # !!! removed read_calib_file

# %%
# This is related to the utils import

def invert_pose(R, T):
    """
    Given the 'sampled pose' (aka H_init), we want CMRNet to predict inv(H_init).
    inv(T*R) will be used as ground truth for the network.
    Args:
        R (mathutils.Euler): Rotation of 'sampled pose'
        T (mathutils.Vector): Translation of 'sampled pose'

    Returns:
        (R_GT, T_GT) = (mathutils.Quaternion, mathutils.Vector)
    """
    R = R.to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(T)
    RT = T * R
    RT.invert_safe()
    T_GT, R_GT, _ = RT.decompose()
    return R_GT.normalized(), T_GT

def quat2mat(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [4x4] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((4, 4), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    mat[3, 3] = 1.
    return mat

def tvector2mat(t):
    """
    Translation vector to homogeneous transformation matrix with identity rotation
    Args:
        t (torch.Tensor): shape=[3], translation vector

    Returns:
        torch.Tensor: [4x4] homogeneous transformation matrix

    """
    assert t.shape == torch.Size([3]), "Not a valid translation"
    mat = torch.eye(4, device=t.device)
    mat[0, 3] = t[0]
    mat[1, 3] = t[1]
    mat[2, 3] = t[2]
    return mat

def rotate_points(PC, R, T=None, inverse=True):
    if T is not None:
        R = R.to_matrix()
        R.resize_4x4()
        T = mathutils.Matrix.Translation(T)
        RT = T*R
    else:
        RT=R.copy()
    if inverse:
        RT.invert_safe()
    RT = torch.tensor(RT, device=PC.device, dtype=torch.float)

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC


def rotate_points_torch(PC, R, T=None, inverse=True):
    if T is not None:
        R = quat2mat(R)
        T = tvector2mat(T)
        RT = torch.mm(T, R)
    else:
        RT = R.clone()
    if inverse:
        RT = RT.inverse()

    if PC.shape[0] == 4:
        PC = torch.mm(RT, PC)
    elif PC.shape[1] == 4:
        PC = torch.mm(RT, PC.t())
        PC = PC.t()
    else:
        raise TypeError("Point cloud must have shape [Nx4] or [4xN] (homogeneous coordinates)")
    return PC

def rotate_forward(PC, R, T=None):
    """
    Transform the point cloud PC, so to have the points 'as seen from' the new
    pose T*R
    Args:
        PC (torch.Tensor): Point Cloud to be transformed, shape [4xN] or [Nx4]
        R (torch.Tensor/mathutils.Euler): can be either:
            * (mathutils.Euler) euler angles of the rotation part, in this case T cannot be None
            * (torch.Tensor shape [4]) quaternion representation of the rotation part, in this case T cannot be None
            * (mathutils.Matrix shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
            * (torch.Tensor shape [4x4]) Rotation matrix,
                in this case it should contains the translation part, and T should be None
         (torch.Tensor/mathutils.Vector): Translation of the new pose, shape [3], or None (depending on R)

    Returns:
        torch.Tensor: Transformed Point Cloud 'as seen from' pose T*R
    """
    if isinstance(R, torch.Tensor):
        return rotate_points_torch(PC, R, T, inverse=True)
    else:
        return rotate_points(PC, R, T, inverse=True)

# %%


class DatasetLidarCamera(Dataset):

    def __init__(self, dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                 max_t=1.5, max_r=20., split='val', device='cpu', val_sequence= 'HFL_CAM_2022-03-15-14-28-46', suf='.png'):  #'HFL_CAM_NEUV_2022-07-07-17-08-02', suf='.png'):
        super(DatasetLidarCamera, self).__init__()
        self.use_reflectance = use_reflectance
        self.maps_folder = ''
        self.device = device
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dir = dataset_dir
        self.transform = transform
        self.split = split
        # self.GTs_R = {}
        # self.GTs_T = {}
        self.GT_cam2hfl = {}
        self.K = {}
        self.suf = suf
        self.all_files = []
        self.sequence_list = ['2021-10-26-13-16-44']# ,'HFL_CAM_2022-03-15-14-28-46'] #,'HFL_CAM_NEUV_2022-07-07-17-08-02']
        #self.sequence_list = ['HFL_CAM_2022-03-15-14-28-46'] #'00', '01', '02']

        ############### Own functions here ###############
        def find_by_header(calib_file, part_header):
            """Finds content of calibration file by header string.

            Args:
                calib_file (str): Path to calibration file.
                part_header (str): Header to calibration data.

            Returns:
                [tuple]: Tuple of np.ndarrays of intrinsics/extrinsics and distortion
                         parameters. If extrnsics distortion returns None.
            """
            with open(calib_file, 'r') as f:
                content = f.readlines()
            #print("content would be all details of calibration")
            full_header = [i for i in content if part_header in i][0]
            print(f'full_header:', full_header)
            print(f'part_header:', part_header)
            idx = content.index(full_header)
            #print(f"content hfl cam idx:",idx)
            content = content[idx:]

            # Distortion parameters
            if 'Intrinsic' in full_header:
                content_dist = content
                print(f'content_dist:', content_dist)
                idx = content_dist.index([i for i in content_dist if 'Dist:' in i][0])
                # print(f"camera Intrinsic idx:", idx)
                content_dist = content_dist[idx+1]
                for r in ['[[', ']]']:
                    content_dist = content_dist.replace(r, '')
                content_dist = content_dist.split()
                dist = np.array([float(i) for i in content_dist])
            else:
                dist = None


            # Calibration Matrices
            idx1 = 1
            idx2 = content.index([i for i in content if ']]' in i][0])
            content = content[idx1:idx2+1]
            
            print(f'Matrices Calib shape:', content)
            #print(f'dimension of Calib matrices:', (idx1,idx2))
        #    return content
            l = list()
            for row in content:
                for r in ['[[', ']]', '[', ']']:
                    row = row.replace(r, '')
                row = row.split()
                l.append([float(i) for i in row])
            return np.array(l), dist
            
        ##################################################

        for seq in self.sequence_list:
            # Read calibration file
            calib_file = os.path.join(dataset_dir, 'sequences', seq, f'calib_{seq}.txt')
            print(f"calib_file found", (calib_file, seq))

            self.GT_cam2hfl[seq], _ = find_by_header(calib_file, 'Tansformation matrix HFL -> Cam:')  #GTpose from cam->HFL (4x4)
            print(f'transformation calibration:', self.GT_cam2hfl[seq].shape)

            self.K[seq], _ = find_by_header(calib_file, 'Camera K_cam:')  # 3x3
            print(f'camera Intrinsic calibration:', self.K[seq].shape)
            #self.GT_cam2hfl[seq], _ = find_by_header(calib_file, 'Transformation (cam -> hfl)')  # GT pose from cam -> HFL (4x4)
            #self.K[seq], _ = find_by_header(calib_file, 'Camera Intrinsic')  # 3x3

            image_path = os.path.join(dataset_dir, 'sequences', seq, 'image_raw') 
            #print(f'image path:', image_path)
           # image_list = os.listdir(os.path.join(dataset_dir, 'sequences', seq, 'image_raw'))
            image_list = os.listdir(image_path)
          #  print(f"image_list:", image_list)
         #   image_list.sort()

            for image_name in image_list:
               # print(f"directory path:", dataset_dir)
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'lidar',
                                                   str(image_name.split('.')[0])+'.bin')):
                    continue
                if not os.path.exists(os.path.join(dataset_dir, 'sequences', seq, 'image_raw',
                                                   str(image_name.split('.')[0])+suf)):
                    continue
                if seq == val_sequence:
                    if split.startswith('val') or split == 'test':
                        self.all_files.append(os.path.join(seq, image_name.split('.')[0]))
                elif (not seq == val_sequence) and split == 'train':
                    self.all_files.append(os.path.join(seq, image_name.split('.')[0]))

        self.val_RT = []
        if split == 'val' or split == 'test':
            val_RT_file = os.path.join(dataset_dir, 'sequences',
                                       f'val_RT_left_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')

            #### Original (modified by Johannes) ####
            # if os.path.exists(val_RT_file):
            #     print(f'VAL SET: Using this file: {val_RT_file}')
            #     df_test_RT = pd.read_csv(val_RT_file, sep=',')
            #     for index, row in df_test_RT.iterrows():
            #         self.val_RT.append(list(row))
            #
            # else:
            #     print(f'VAL SET - Not found: {val_RT_file}')
            #     print("Generating a new one")
            #     val_RT_file = open(val_RT_file, 'w')
            #     val_RT_file = csv.writer(val_RT_file, delimiter=',')
            #     val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
            #     for i in range(len(self.all_files)):
            #         rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
            #         roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
            #         rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
            #         transl_x = np.random.uniform(-max_t, max_t)
            #         transl_y = np.random.uniform(-max_t, max_t)
            #         transl_z = np.random.uniform(-max_t, max_t)
            #         val_RT_file.writerow([i, transl_x, transl_y, transl_z,
            #                                rotx, roty, rotz])
            #         self.val_RT.append([float(i), float(transl_x), float(transl_y), float(transl_z),
            #                              float(rotx), float(roty), float(rotz)])
            #### ------------------------------- ####

            if os.path.exists(val_RT_file):
                os.remove(val_RT_file)
                print(f'\nVAL SET: File {val_RT_file} deleted')

            print(f'Generating a new VAL SET {val_RT_file}')
            val_RT_file = open(val_RT_file, 'w')
            val_RT_file = csv.writer(val_RT_file, delimiter=',')
            val_RT_file.writerow(['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz'])
            for i in range(len(self.all_files)):
                rotz = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                roty = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                rotx = np.random.uniform(-max_r, max_r) * (3.141592 / 180.0)
                transl_x = np.random.uniform(-max_t, max_t)
                transl_y = np.random.uniform(-max_t, max_t)
                transl_z = np.random.uniform(-max_t, max_t)
                val_RT_file.writerow([i, transl_x, transl_y, transl_z,
                                        rotx, roty, rotz])
                self.val_RT.append([float(i), float(transl_x), float(transl_y), float(transl_z),
                                        float(rotx), float(roty), float(rotz)])

            assert len(self.val_RT) == len(self.all_files), "Something wrong with test RTs"


    def custom_transform(self, rgb, img_rotation=0., flip=False):
        to_tensor = transforms.ToTensor()
        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        #rgb = crop(rgb)
        if self.split == 'train':
            color_transform = transforms.ColorJitter(0.1, 0.1, 0.1)
            rgb = color_transform(rgb)
            if flip:
                rgb = TTF.hflip(rgb)
            rgb = TTF.rotate(rgb, img_rotation)
            #io.imshow(np.array(rgb))
            #io.show()

        rgb = to_tensor(rgb)
        rgb = normalization(rgb)
        return rgb

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        print(f'reading lidar scans')
        item = self.all_files[idx]
        seq = str(item.split('/')[0])
        rgb_name = str(item.split('/')[1])
        img_path = os.path.join(self.root_dir, 'sequences', seq, 'image_raw', rgb_name+self.suf)
        lidar_path = os.path.join(self.root_dir, 'sequences', seq, 'lidar', rgb_name+'.bin')
        #print(f'lidar files path:', lidar_path)
        lidar_scan = np.fromfile(lidar_path, dtype=np.float32)
        pc = lidar_scan.reshape((-1, 4))
        valid_indices = pc[:, 0] < -3.
        valid_indices = valid_indices | (pc[:, 0] > 3.)
        valid_indices = valid_indices | (pc[:, 1] < -3.)
        valid_indices = valid_indices | (pc[:, 1] > 3.)
        pc = pc[valid_indices].copy()
        pc_org = torch.from_numpy(pc.astype(np.float32))
       # print(f'the lidar scan pc_org', pc_org)

        RT = self.GT_cam2hfl[seq].astype(np.float32)
        RT = RT.reshape(4,4)
        print('Shape of RT:', RT.shape)

        if pc_org.shape[1] == 4 or pc_org.shape[1] == 3:
            pc_org = pc_org.t()
        if pc_org.shape[0] == 3:
            homogeneous = torch.ones(pc_org.shape[1]).unsqueeze(0)
            pc_org = torch.cat((pc_org, homogeneous), 0)
        elif pc_org.shape[0] == 4:
            if not torch.all(pc_org[3, :] == 1.):
                pc_org[3, :] = 1.
        else:
            raise TypeError("Wrong PointCloud shape")
       
        #print(f'shape of pc_org',pc_org.numpy())
        pc_rot = np.matmul(RT, pc_org.numpy())
        #pc_rot = np.matmul(RT, pc_org)
        pc_rot = pc_rot.astype(np.float32).copy()
        pc_in = torch.from_numpy(pc_rot)

        h_mirror = False

        img = Image.open(img_path)
        img_rotation = 0.

        try:
            img = self.custom_transform(img, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        if self.split == 'train':
            R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        if self.split == 'train':
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, self.max_t)
            # transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
        else:
            initial_RT = self.val_RT[idx]
            rotz = initial_RT[6]
            roty = initial_RT[5]
            rotx = initial_RT[4]
            transl_x = initial_RT[1]
            transl_y = initial_RT[2]
            transl_z = initial_RT[3]

        # train different paramters at every epoch
        # test initialization, every epoch same parameters
        R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
        T = mathutils.Vector((transl_x, transl_y, transl_z))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        #io.imshow(depth_img.numpy(), cmap='jet')
        #io.show()
        calib = self.K[seq]
        if h_mirror:
            calib[2] = (img.shape[2] / 2)*2 - calib[2]

        if self.split == 'test':
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'seq': int(seq), 'img_path': img_path,
                      'rgb_name': rgb_name + '.png', 'item': item, 'extrin': RT,
                      'initial_RT': initial_RT}
        else:
            sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib,
                      'tr_error': T, 'rot_error': R, 'seq': int(seq),
                      'rgb_name': rgb_name, 'item': item, 'extrin': RT}

        return sample


# %%
#### TESTING ####
#dataset_dir = '/home/aditya/HFL_data/HFL_CAM_NEUV_2022-07-07/dataset/'
dataset_dir = '/home/aditya/HFL_data/newdata/dataset/'
data = DatasetLidarCamera(dataset_dir, transform=None, augmentation=False, use_reflectance=False,
                            max_t=1.5, max_r=20., split='train', device='cpu', val_sequence= '00', suf='.png')

#print(f'Dataloading done',data)
#data = DatasetLidarCamera(dataset_dir, transform=None, augmentation=False, use_reflectance=False,
#                            max_t=1.5, max_r=20., split='test', device='cpu', val_sequence='00', suf='.png')
# %%



