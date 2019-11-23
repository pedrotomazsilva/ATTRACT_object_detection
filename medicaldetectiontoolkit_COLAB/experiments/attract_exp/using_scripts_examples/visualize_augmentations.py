from datautils.data import DataGenerator
import numpy as np
import skimage.io as io
from image_slicer import slice_cubes, slicer
import matplotlib.pyplot as plt
from imutils.image_processing import draw_mask_bounds

# Parameters
data_configs = {'dim': (3,128,128,32),
                'mask_dim':(1,128,128,32),
                'batch_size': 4,
                'shuffle': False}

# Generators
data_aug_dict = {

    'do_elastic_deform':False,
    'alpha':(0., 1000.),
    'sigma':(10., 13.),
    'do_rotation':False,
    'angle_x':(0, 0),
    'angle_y':(0, 0),
    'angle_z':(0, 2*np.pi),
    'do_scale':False,
    'scale':(0.75, 0.75),
    'border_mode_data':'constant',
    'border_cval_data':0,
    'order_data':3,
    'border_mode_seg':'constant',
    'border_cval_seg':0,
    'order_seg':0,
    'random_crop':False,
    'p_el_per_sample':0.5,
    'p_scale_per_sample':1,
    'p_rot_per_sample':1
}

train_generator = DataGenerator(partition='val', configs=data_configs, data_aug_dict=data_aug_dict)
imgs, mask = train_generator.__getitem__(0)

mask1 = np.concatenate((mask[0],mask[0],mask[0]),axis=0)
#mask1 = np.transpose(mask1,(1,2,0,3))

cube_mask_dir = 'D:/bora comprimir/data/val/mask/mask_40_cube.npy'
mask1_gt = np.load(cube_mask_dir)
mask1_gt = np.concatenate((mask1_gt,mask1_gt,mask1_gt),axis=0)
#mask1_gt = np.transpose(mask1_gt,(1,2,0,3))


print(np.array_equal(mask1_gt, mask1))
cube_1 = np.expand_dims(imgs[0],axis=0)
cube_1[0,2,:,:,:]=0
cube_1[cube_1>0.02]=0.5

bounds = draw_mask_bounds(cube_1,np.expand_dims(mask1,axis=0))
slicer(np.transpose(bounds[0],(1,2,0,3)))



'''
cube_mask = np.zeros((mask.shape[0], 3,128,128,32))
cube_mask[:,0,:,:,:]=np.squeeze(mask)
cube_mask[:,1,:,:,:]=np.squeeze(mask)
cube_mask[:,2,:,:,:]=np.squeeze(mask)

imgs[:,2,:,:,:]=0
imgs[imgs>0.02]=0.5

slice_cubes(imgs,cube_mask)

'''
