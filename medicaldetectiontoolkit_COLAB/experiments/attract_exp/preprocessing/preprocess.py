import numpy as np
import skimage.io as io
import os
import scipy.io as sio
import matplotlib.pyplot as plt


def rescale_img_values(img, max=None, min=None):

    if max == None:
        max = img.max()
    if min == None:
        min = img.min()

    if max != min:
        rescale_factor = 1 / (max - min)
        img = (img - min) * rescale_factor

    return img


def normalize_img_values(img, normalize_each_slice=True):

    if normalize_each_slice:
        for i in range(img.shape[0]):
            if np.std(img[i])!=0:
                img[i] = (img[i]-np.mean(img[i]))/np.std(img[i])
    else:
        std = np.std(img)
        mean = np.mean(img)
        img -= mean
        if np.std(img) != 0:
            img /= std

    return img


def organize_cube_data_from_cube_files(in_data_dir, out_data_dir, z_size=32):

    cube_number = [s for s in in_data_dir.split(sep='_') if s.isdigit()][-1]

    for i in range(min(len(os.listdir(in_data_dir)), z_size)):
        img_slice = np.expand_dims(
            io.imread(in_data_dir + '/slice_' + str(i+1) + '/slice_' + str(i+1) + '_img.png'), axis=0)
        mask_slice = np.expand_dims(
            io.imread(in_data_dir + '/slice_' + str(i+1) + '/slice_' + str(i+1) + '_mask.png', as_gray=True),axis=0)

        if i==0:
            img_cube=img_slice
            mask_cube=mask_slice
        else:
            img_cube = np.concatenate((img_cube, img_slice))
            mask_cube = np.concatenate((mask_cube, mask_slice))


    # Normalize and rescale img and mask values

    img_cube = rescale_img_values(normalize_img_values(img_cube.astype(dtype=np.float64)))
    mask_cube = rescale_img_values(mask_cube.astype(dtype=np.float64))
    img_cube = img_cube.transpose((3,1,2,0)) #(n_channels, X, Y, Z)
    mask_cube = np.expand_dims(mask_cube.transpose((1, 2, 0)), axis=0) #(n_classes,X,Y,Z)


    np.save(out_data_dir+'/img/'+cube_number+'_cube.npy', img_cube)
    np.save(out_data_dir + '/mask/mask_' + cube_number + '_cube.npy', mask_cube)



def organize_cube_data(out_data_dir, cubes, cubes_mask):


    for cube_number in range(cubes.shape[0]):
        # Normalize and rescale img and mask values
        img_cube = cubes[cube_number]
        mask_cube = cubes_mask[cube_number]

        img_cube = rescale_img_values(normalize_img_values(img_cube.astype(dtype=np.float64)))
        mask_cube = rescale_img_values(mask_cube.astype(dtype=np.float64))
        img_cube = img_cube.transpose((2,0,1,3)) #(n_channels, X, Y, Z)
        mask_cube = mask_cube.transpose((2,0,1,3)) #(n_classes,X,Y,Z)


        np.save(out_data_dir+'/img/'+str(cube_number)+'_cube.npy', img_cube)
        np.save(out_data_dir + '/mask/mask_' + str(cube_number) + '_cube.npy', mask_cube)



def divide_into_cubes(img,mask,cubes_size, no_overlap=True):


    cubes_per_x = img.shape[0] // cubes_size[0]
    cubes_per_y = img.shape[1] // cubes_size[1]
    cubes_per_z = img.shape[3] // cubes_size[2]

    if no_overlap == True:
        additional_x_cube = 0
        additional_y_cube = 0
        additional_z_cube = 0
    else:
        additional_x_cube = min(1, mask.shape[0] % cubes_size[0])
        additional_y_cube = min(1, mask.shape[1] % cubes_size[1])
        additional_z_cube = min(1, mask.shape[3] % cubes_size[2])

    n_cubes = cubes_per_x * cubes_per_y * cubes_per_z + additional_x_cube * cubes_per_y * cubes_per_z + \
              additional_y_cube * cubes_per_x * cubes_per_z + additional_z_cube * cubes_per_x * cubes_per_y

    if cubes_per_x * cubes_per_y * cubes_per_z == 0:
        print('Cubes are too big for img size')

    if len(mask.shape)==3:
        mask = np.expand_dims(mask, 2)

    cubes_mask = np.zeros((n_cubes, cubes_size[0], cubes_size[1], mask.shape[2], cubes_size[2]))
    cubes = np.zeros((n_cubes, cubes_size[0], cubes_size[1], img.shape[2], cubes_size[2]))


    i = 0
    for n_cubes_x in range(cubes_per_x + additional_x_cube):
        for n_cubes_y in range(cubes_per_y + additional_y_cube):
            for n_cubes_z in range(cubes_per_z + additional_z_cube):

                if n_cubes_x == cubes_per_x: #additional cube
                    x_min = mask.shape[0] - cubes_size[0]
                    x_max = mask.shape[0]
                else:
                    x_min = n_cubes_x * cubes_size[0]
                    x_max = (n_cubes_x+1) * cubes_size[0]


                if n_cubes_y == cubes_per_y: #additional cube
                    y_min = mask.shape[1] - cubes_size[1]
                    y_max = mask.shape[1]
                else:
                    y_min = n_cubes_y * cubes_size[1]
                    y_max = (n_cubes_y+1) * cubes_size[1]


                if n_cubes_z == cubes_per_z: #additional cube
                    z_min = mask.shape[3] - cubes_size[2]
                    z_max = mask.shape[3]
                else:
                    z_min = n_cubes_z * cubes_size[2]
                    z_max = (n_cubes_z+1) * cubes_size[2]


                cubes_mask[i,:,:,:,:] = mask[x_min:x_max, y_min:y_max, :, z_min:z_max]
                cubes[i,:,:,:,:] = img[x_min:x_max, y_min:y_max, :, z_min:z_max]

                i += 1

    return cubes, cubes_mask


def merge_masks(golgi_dir, nucleus_dir, out_dir):

    for i, nucleus_mask_name in enumerate(os.listdir(nucleus_dir)):
        mask_cube_golgi = np.load(golgi_dir+nucleus_mask_name)
        mask_cube_nucleus = np.load(nucleus_dir + nucleus_mask_name)
        mask_cube = np.concatenate((mask_cube_golgi,mask_cube_nucleus),axis=0)
        np.save(out_dir+nucleus_mask_name,mask_cube)



