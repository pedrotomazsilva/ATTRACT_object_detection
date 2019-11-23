import os
import numpy as np
from image_utilities.image_slicer import slicer_editor, slicer



def create_gt_mask(set_path):

    edited_masks_path = set_path+'/edited_masks/'

    try:
        os.mkdir(edited_masks_path)
    except:
        pass

    for img_path in os.listdir(set_path + '/img/'):

        img = np.load(set_path + '/img/' + img_path)
        img = np.transpose(img, [1,2,0,3])
        new_mask_path = edited_masks_path + 'mask_' + img_path

        mask_shape = (img.shape[0], img.shape[1], img.shape[3])
        slicer_editor(img, np.zeros(mask_shape), new_mask_path)


set_path = 'C:/Users/pedro/OneDrive - Universidade de Lisboa/Projetos/' \
           'Projetos Python/ATTRACT/data/train'

create_gt_mask(set_path)

mask = np.load('D:/ATTRACT/cubes/mask_c_P6_AVM/mask_129_cube.npy')
mask = np.transpose(mask, (1,2,0,3))
mask = np.concatenate((mask,np.zeros((128,128,1,32))),axis=2)
slicer(mask)



