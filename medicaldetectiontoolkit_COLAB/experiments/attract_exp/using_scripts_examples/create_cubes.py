from image_utilities.image_processing import split_image_into_cubes, save_cubes
from image_utilities.image_reading import read_image
import numpy as np


image_path = 'C:/Users/pedro/Desktop/images/czi/P6_ICAM2.czi'
bbox_coords_path = 'C:/Users/pedro/Desktop/P6_coordinates.npy'

out_data_dir_cubes='C:/Users/pedro/Desktop/data/train/img'
out_data_dir_bboxes='C:/Users/pedro/Desktop/data/train/bboxes'


image = read_image(image_path, rgb_order='rgb')

bboxes_coords = np.load(bbox_coords_path,allow_pickle=True)
bboxes_coords = np.delete(bboxes_coords, np.where(bboxes_coords==None)[0],axis=0) #delete possible None rows


cubes, cubes_bboxes = split_image_into_cubes(image, cubes_size=(256,256,32), bboxes_coords=bboxes_coords,
                                             bounding_box=True, no_overlap=True)

save_cubes(out_data_dir_cubes=out_data_dir_cubes, out_data_dir_bbox=out_data_dir_bboxes,
           cubes=cubes, cubes_bbox=cubes_bboxes)

