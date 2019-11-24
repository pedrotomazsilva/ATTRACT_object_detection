from image_utilities.image_processing import split_image_into_cubes
from image_utilities.image_reading import read_image
from image_utilities.image_slicer import slicer, slice_cubes
import numpy as np


image_path = 'C:/Users/pedro/Desktop/images/czi/P6_ICAM2.czi'
bbox_coords_path = 'C:/Users/pedro/Desktop/P6_coordinates.npy'


image = read_image(image_path, rgb_order='rgb')

bboxes_coords = np.load(bbox_coords_path,allow_pickle=True)
bboxes_coords = np.delete(bboxes_coords, np.where(bboxes_coords==None)[0],axis=0) #delete possible None rows


cubes, cubes_bboxes = split_image_into_cubes(image, cubes_size=(256,256,32), bboxes_coords=bboxes_coords,
                                             bounding_box=True, no_overlap=True)

#You can see all cubes using this command by ckicking in the plot to change from cube to cube
slicer(cubes, channels_first=True, slice_cubes=True)

#You can also view several cubes displayed in one plot at a time by using the following:
slice_cubes(cubes,images_per_figure = 4, cubes_mask=None)


