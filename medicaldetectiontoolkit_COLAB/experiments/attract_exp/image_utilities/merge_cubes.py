import numpy as np
import skimage.io as io
import scipy.io as sio
from image_utilities.image_slicer import slicer

def merge_cube_preds(cubes, img_size):

    full_mask = np.zeros(img_size)
    cubes = np.squeeze(cubes)

    cube_no = 0
    for n_x in range(int(img_size[0]//cubes.shape[1])):
        for n_y in range(int(img_size[1]//cubes.shape[2])):
            for n_z in range(int(img_size[2]//cubes.shape[3])):
                full_mask[
                (n_x * cubes.shape[1]):((n_x + 1) * cubes.shape[1]),
                (n_y * cubes.shape[2]):((n_y + 1) * cubes.shape[2]),
                (n_z * cubes.shape[3]):((n_z + 1) * cubes.shape[3])] = cubes[cube_no,:,:,:]
                cube_no += 1

    return np.expand_dims(full_mask,2)


def binarize_prediction_mask(pred_mask, threshold=0.5, avg_threshold=False):

    if avg_threshold:
        threshold = np.mean(pred_mask[:])

    pred_mask[pred_mask>threshold]=1
    pred_mask[pred_mask<=threshold]=0

    return pred_mask

cubes = np.load('C:/Users/pedro/PycharmProjects/ATTRACT/data/test_P6_SF/preds/preds_test.npy')
img = sio.loadmat('D:/iMM_Projeto/código/variables/P6_AVM/P6_AVM_double.mat')['img_3d_double']
img_shape = (img.shape[0], img.shape[1], 32)
full_mask = merge_cube_preds(cubes,img_shape)
np.save('C:/Users/pedro/PycharmProjects/ATTRACT/data/test/preds/preds_full.npy', full_mask)
#full_mask = np.load('C:/Users/pedro/PycharmProjects/ATTRACT/data/test/preds/preds_full.npy')

mask_dict = {
    'mask':full_mask
}
sio.savemat('D:/iMM_Projeto/código/P6_AVM_pred_mask.mat', mask_dict)

cube = cubes[1]
full = np.zeros((cube.shape[1], cube.shape[2], 3, cube.shape[3]))
full[:,:,0,:]=cube
full[:,:,1,:]=cube
full[:,:,2,:]=cube
slicer(full)