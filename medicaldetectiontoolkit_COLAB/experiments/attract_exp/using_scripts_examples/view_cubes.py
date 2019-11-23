from image_utilities.image_slicer import slicer
from image_utilities.image_processing import impose_bbox_cuboids
import numpy as np

out_data_dir_cubes='C:/Users/pedro/Desktop/medicaldetectiontoolkit_COLAB_data/experiments/attract_exp/data/train/img'
out_data_dir_bboxes='C:/Users/pedro/Desktop/medicaldetectiontoolkit_COLAB_data/experiments/attract_exp/data/train/bboxes'


cube = np.load(out_data_dir_cubes+'/0_cube.npy')
bbox_coords = np.load(out_data_dir_bboxes+'/0_cube_bbox.npy').astype(np.int)
cube = impose_bbox_cuboids(cube,bbox_coords)
slicer(cube, channels_first=True, slice_cubes=False)

