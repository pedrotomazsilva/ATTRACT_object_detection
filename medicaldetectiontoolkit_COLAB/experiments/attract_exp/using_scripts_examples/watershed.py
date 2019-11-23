import numpy as np
import scipy.io as sio
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage.segmentation import random_walker
from imutils.image_slicer import slicer
from skimage.color import label2rgb
from imutils.image_processing import to_rgb_white

mask_path = 'D:/iMM_Projeto/código/variables/P6/P6_final_mask_nucleus.mat'


mask_dict = sio.loadmat(mask_path)
mask = mask_dict['final_mask_nucleus']
mask = mask.astype(np.float64)
distance = ndi.distance_transform_edt(mask)

#distance_max = np.max(distance)
#slicer(to_rgb_white(distance/distance_max), slice_cubes=False,channels_first=False)

local_maxi = peak_local_max(distance, labels=mask,
                            footprint=np.ones((10, 10, 10)),
                            indices=False)

#slicer(to_rgb_white(local_maxi.astype(np.float64)), slice_cubes=False,channels_first=False)
markers = ndi.label(local_maxi)[0]

#slicer(to_rgb_white(markers/np.max(markers)), slice_cubes=False,channels_first=False)
labels = watershed(-distance, markers=markers, mask=mask)

labels_rgb = np.transpose(label2rgb(labels),(0,1,3,2))
labels_rgb = labels_rgb * to_rgb_white(labels)
slicer(labels_rgb)

