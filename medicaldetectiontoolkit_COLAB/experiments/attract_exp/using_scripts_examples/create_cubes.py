from image_utilities.image_processing import split_image_into_cubes, save_cubes
import scipy.io as sio
import numpy as np


centroid_path = 'C:/Users/pedro/OneDrive - Universidade de Lisboa' \
                '/Projetos/Projetos Python/ATTRACT/Marie_assignments/P6_SF_all_data.csv' # P6_ICAM2_all_data

image_path = 'D:/iMM_Projeto/código/variables/P6/P6_double.mat'

out_data_dir_cubes='C:/Users/pedro/Desktop/data/train/img'
out_data_dir_bboxes='C:/Users/pedro/Desktop/data/train/bboxes'


image_dict = sio.loadmat(image_path)
image = image_dict['img_3d_double']

bboxes_coords = np.load('data/bbox_coordinates_P6.npy',allow_pickle=True)
bboxes_coords = np.delete(bboxes_coords, 2,0) #delete nan row

cubes, cubes_bboxes = split_image_into_cubes(image, cubes_size=(256,256,32), bboxes_coords=bboxes_coords,
                                             bounding_box=True, no_overlap=True)



save_cubes(out_data_dir_cubes=out_data_dir_cubes, out_data_dir_bbox=out_data_dir_bboxes,
           cubes=cubes, cubes_bbox=cubes_bboxes)

