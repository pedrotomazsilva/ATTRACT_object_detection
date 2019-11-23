from imutils.image_slicer import slicer
from imutils.image_processing import impose_bbox_cuboids, create_nucleus_cuboids
from datautils.data import read_all_cubes
import numpy as np

out_data_dir_cubes='C:/Users/pedro/Desktop/data/train/img'
out_data_dir_bboxes='C:/Users/pedro/Desktop/data/train/bboxes'

#cubes = read_all_cubes(cubes_path=out_data_dir_cubes)
cube = np.load(out_data_dir_cubes+'/0_cube.npy')
bbox_coords = np.load(out_data_dir_bboxes+'/0_cube_bbox.npy').astype(np.int)
#bbox_coords[bbox_coords==0]=1
cube = impose_bbox_cuboids(cube,bbox_coords)
slicer(cube, channels_first=True, slice_cubes=False)