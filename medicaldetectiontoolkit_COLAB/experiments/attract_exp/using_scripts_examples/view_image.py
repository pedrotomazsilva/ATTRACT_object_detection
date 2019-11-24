from image_utilities.image_slicer import slicer
from image_utilities.image_processing import impose_bbox_cuboids
from image_utilities.image_reading import read_image
import numpy as np


image_path = 'C:/Users/pedro/Desktop/images/czi/P6_ICAM2.czi'
bbox_coords_path = 'C:/Users/pedro/Desktop/P6_coordinates.npy' #only if you have bounding boxes

image = read_image(image_path, rgb_order='rgb')

bboxes_coords = np.load(bbox_coords_path,allow_pickle=True)
bboxes_coords = np.delete(bboxes_coords, np.where(bboxes_coords==None)[0],axis=0)#delete possible None rows
image = impose_bbox_cuboids(image,bboxes_coords) #view bounding boxes (if those exist)
slicer(image, channels_first=True, slice_cubes=False)

