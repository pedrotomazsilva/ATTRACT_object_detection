from imutils.image_slicer import slicer
from imutils.image_processing import impose_mask_on_cubes, adjust_img_adj, to_rgb_white, draw_mask_bounds
from datautils.data import read_all_cubes
from evalutils.model_evaluation import plot_trainining_evolution
import numpy as np

preds_path = 'preds_val.npy'

ground_truth_dir = 'D:/bora comprimir 2/data/val/mask'


image_dir = 'D:/bora comprimir 2/data/val/img_adj'


prediction_cubes = np.load(preds_path)
image_cubes = read_all_cubes(image_dir)
#ground_truth_cubes = read_all_cubes(ground_truth_dir,mask=True)

#imposed_cubes = impose_mask_on_cubes(adjust_img_adj(image_cubes), prediction_cubes,mode='maximum_color_per_pixel')
imposed_cubes = draw_mask_bounds(adjust_img_adj(image_cubes),prediction_cubes,component=1)

#plot_trainining_evolution('training.log',loss='binary_crossentropy',multi_class=False)
plot_trainining_evolution('training.log',multi_class=True, loss='dice')
slicer(imposed_cubes, slice_cubes=True, channels_first=True)



