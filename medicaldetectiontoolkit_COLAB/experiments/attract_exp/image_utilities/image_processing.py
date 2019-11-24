import numpy as np
import skimage.filters as filters


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

def draw_mask_bounds(cubes,cubes_mask,ground_truth=None,component=0):

    cubes_wbounds = np.copy(cubes)

    for cube_n, cube_mask in enumerate(cubes_mask):
        for n_slice in range(cube_mask.shape[-1]):

            if ground_truth is not None:
                bounds = filters.sobel(ground_truth[cube_n, component, :, :, n_slice])
                #cubes_wbounds[cube_n, 0, :, :, n_slice] = np.maximum(bounds, cubes_wbounds[cube_n, 0, :, :, n_slice])
                #cubes_wbounds[cube_n, 1, :, :, n_slice] = np.maximum(bounds, cubes_wbounds[cube_n, 1, :, :, n_slice])
                cubes_wbounds[cube_n, 2, :, :, n_slice] = np.maximum(bounds, cubes_wbounds[cube_n, 2, :, :, n_slice])

            bounds = filters.sobel(cube_mask[component, :, :, n_slice])
            cubes_wbounds[cube_n, 0, :, :, n_slice] = np.maximum(bounds, cubes_wbounds[cube_n, 0, :, :, n_slice])
            cubes_wbounds[cube_n, 1, :, :, n_slice] = np.maximum(bounds, cubes_wbounds[cube_n, 1, :, :, n_slice])
            cubes_wbounds[cube_n, 2, :, :, n_slice] = np.maximum(bounds, cubes_wbounds[cube_n, 2, :, :, n_slice])

    return cubes_wbounds

def threshold_mask(cubes_mask,threshold=0.5):

    cubes_mask[cubes_mask > threshold] = 1
    cubes_mask[cubes_mask <= threshold] = 0

    return cubes_mask

def to_rgb_white(one_channel_white_3d_img):

    if len(one_channel_white_3d_img.shape)==5: #shape (n_cubes,channel_gray,x,y,z)
        axis=1
    else: #shape (x,y,z)
        one_channel_white_3d_img = np.expand_dims(one_channel_white_3d_img, axis=2)
        axis=2


    white_img = np.concatenate((one_channel_white_3d_img,one_channel_white_3d_img,
                                one_channel_white_3d_img),axis=axis)
    return white_img

def centroids_to_gaussian_regions(centroids, mask_shape, gaussian_ball_std, gaussian_ball_max_radius):
    """
    Draws gaussian balls for each centroid
    :param centroids: numpy array (n_pairs, xyz nucleus, xyz corresponding golgi)
    :param mask_shape: tuple with the shape of the black image in which the gaussian regions
    will be drawn
    :param gaussian_ball_std: standard deviation of the gaussian region centered on each centroid
    :param gaussian_ball_max_radius: max radius of the gaussian ball
    :return: image with gaussian regions where the pixels are between 0 and 1 where 1 is the value
    for the centroid. One image for the nucleus and another for the golgis
    """

    mask_nucleus = np.zeros(mask_shape)
    mask_golgi = np.zeros(mask_shape)

    gaussian_ball = make_gaussian_ball(gaussian_ball_std, gaussian_ball_max_radius)

    for pairs in centroids:
        mask_nucleus = add_gaussian_ball(mask_nucleus,gaussian_ball,pairs[1],pairs[0],pairs[2])
        mask_golgi = add_gaussian_ball(mask_golgi,gaussian_ball,pairs[4],pairs[3],pairs[5])

    return mask_nucleus, mask_golgi

def add_gaussian_ball(mask,gaussian_ball, x,y,z):

    x_ball_size = gaussian_ball.shape[0]
    y_ball_size = gaussian_ball.shape[1]
    z_ball_size = gaussian_ball.shape[2]

    v_range1 = slice(max(0, x - int(np.floor(x_ball_size/2))),
                     max(min(x+int(np.floor(x_ball_size/2)+1), mask.shape[0]), 0))
    h_range1 = slice(max(0, y - int(np.floor(y_ball_size/2))),
                     max(min(y+int(np.floor(y_ball_size/2)+1), mask.shape[1]), 0))
    z_range1 = slice(max(0, z - int(np.floor(z_ball_size/2))),
                     max(min(z+int(np.floor(z_ball_size/2)+1), mask.shape[2]), 0))

    v_range2 = slice(max(0, int(np.floor(x_ball_size/2))-x),
                     min(mask.shape[0] - (x+1) + int(np.floor(x_ball_size / 2)+1), x_ball_size))
    h_range2 = slice(max(0, int(np.floor(y_ball_size/2))-y),
                     min(mask.shape[1] - (y+1) + int(np.floor(y_ball_size / 2)+1), y_ball_size))
    z_range2 = slice(max(0, int(np.floor(z_ball_size/2))-z),
                     min(mask.shape[2] - (z+1) + int(np.floor(z_ball_size/2)+1), z_ball_size))

    mask[v_range1, h_range1, z_range1] += gaussian_ball[v_range2, h_range2, z_range2]

    return mask

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def make_gaussian_ball(gaussian_ball_std, max_radius):

    gaussian_ball = np.zeros((2*max_radius+1, 2*max_radius+1, 2*max_radius+1))


    r_values = []

    for time in range(2):

        for i in range(2*max_radius+1):
            for j in range(2*max_radius+1):
                for k in range(2*max_radius+1):

                    if time == 0:
                        r_values.append((i-max_radius)**2+(j-max_radius)**2+(k-max_radius)**2)
                    else:
                        r_value = (i - max_radius) ** 2 + (j - max_radius) ** 2 + \
                                  (k - max_radius) ** 2
                        if (r_value <= max_radius ** 2):
                            gaussian_ball[i, j, k] = p_values[np.where(r_values==r_value)]

        if time == 0:
            r_values = np.unique(np.array(r_values))
            p_values = np.arange(r_values.size)
            p_values = gaussian(p_values, 0, gaussian_ball_std)

    return gaussian_ball

def impose_mask(img,mask):

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[3]):
                if mask[i,j,k]>0:
                    img[i,j,:,k]=mask[i,j,k]

    return img

def impose_mask_on_cubes(cubes_img, cubes_mask, mode='greater_than_zero_ppixel'):

    for cube_i in range(cubes_img.shape[0]):
        for i in range(cubes_img.shape[2]):
            for j in range(cubes_img.shape[3]):
                for k in range(cubes_img.shape[4]):
                    if mode == 'maximum_color_per_pixel':
                        if cubes_mask[cube_i,0,i, j, k] > max(cubes_img[cube_i, 0, i, j, k],
                                                              cubes_img[cube_i, 1, i, j, k],
                                                              cubes_img[cube_i, 2, i, j, k]):

                            cubes_img[cube_i, :, i, j, k] = cubes_mask[cube_i,0,i, j, k]
                    else:
                        if cubes_mask[cube_i, 0, i, j, k] > 0:
                            cubes_img[cube_i, :, i, j, k] = cubes_mask[cube_i, 0, i, j, k]

    return cubes_img

def adjust_img_adj(cubes_adj):

    if cubes_adj.shape == 5:
        for i, cubinho in enumerate(cubes_adj):
            cubinho[2, :, :, :] = 0
            cubinho = cubinho * 4
            cubes_adj[i]=cubinho
    else:
        cubes_adj[2, :, :, :] = 0
        cubes_adj = cubes_adj * 4

    return cubes_adj

def get_bboxes_coord(bboxes_coords, x_min, x_max, y_min, y_max, z_min, z_max):
    """

    :param centroids: numpy array (n_pairs, )
    :return:
    """
    cube_bboxes = None

    for pair in bboxes_coords:

        is_pt1_in_cube = False
        is_pt2_in_cube = False

        if (pair[0] >= x_min and pair[0] < x_max) and (pair[1] >= y_min and pair[1] < y_max) \
                and (pair[4] >= z_min or pair[4] < z_max):
            is_pt1_in_cube = True

        if (pair[2] >= x_min and pair[2] < x_max) and (pair[3] >= y_min and pair[3] < y_max) \
                and (pair[5] >= z_min or pair[5] < z_max):
            is_pt2_in_cube = True

        if is_pt1_in_cube or is_pt2_in_cube:


            if is_pt1_in_cube and not is_pt2_in_cube:

                if pair[2]-x_min<0:
                    pair[2]=0
                if pair[2]-x_max>0:
                    pair[2]=x_max-x_min-1

                if pair[3]-y_min<0:
                    pair[3]=0
                if pair[3]-y_max>0:
                    pair[3]=y_max-y_min-1

                if pair[5]-z_min<0:
                    pair[5]=0
                if pair[5]-z_max>0:
                    pair[5]=z_max-z_min-1

            if is_pt2_in_cube and not is_pt1_in_cube:

                if pair[0] - x_min < 0:
                    pair[0] = 0
                if pair[0] - x_max > 0:
                    pair[0] = x_max - x_min - 1

                if pair[1] - y_min < 0:
                    pair[1] = 0
                if pair[1] - y_max > 0:
                    pair[1] = y_max - y_min - 1

                if pair[4] - z_min < 0:
                    pair[4] = 0
                if pair[4] - z_max > 0:
                    pair[4] = z_max - z_min - 1


            x_i = int(pair[0] - x_min)
            y_i = int(pair[1] - y_min)
            z_i = int(pair[4] - z_min)

            x_j = int(pair[2] - x_min)
            y_j = int(pair[3] - y_min)
            z_j = int(pair[5] - z_min)

            coor_array = np.expand_dims(np.array([min(x_i,x_j), min(y_i, y_j),
                                                  max(x_i,x_j), max(y_i,y_j), min(z_i,z_j), max(z_i,z_j)]), axis=0)

            if cube_bboxes is None:
                cube_bboxes = coor_array
            else:
                cube_bboxes = np.concatenate((cube_bboxes, coor_array), axis=0)

    return cube_bboxes

def split_image_into_cubes(img, cubes_size, bboxes_coords=None, bounding_box=False, no_overlap=True):
    """
    :param img: shape (channels, x,y,z)
    :param cubes_size: shape of the cubes ex: (256,256,32)
    :param bounding_box: return the bounding box coordinates for each roi in the cube
    :param no_overlap: Decide if we ant overlapping cubes to account fot the whole image
    (This only makes sense when one or more components of the image size are not dividable
    by the cubes size
    :return:
    image cubes: (num_cubes, channels, x, y, z)
    bounding box coordinates for each pair: list (num_cubes, n_rois, x, y, z, class)
    where roi stands for region of interest (can be region for each pair, nucleus, etc...)

    """

    img = np.transpose(img, (1,2,0,3))
    cubes_per_x = img.shape[0] // cubes_size[0]
    cubes_per_y = img.shape[1] // cubes_size[1]
    cubes_per_z = img.shape[3] // cubes_size[2]

    if no_overlap == True:
        additional_x_cube = 0
        additional_y_cube = 0
        additional_z_cube = 0
    else:
        additional_x_cube = min(1, img.shape[0] % cubes_size[0])
        additional_y_cube = min(1, img.shape[1] % cubes_size[1])
        additional_z_cube = min(1, img.shape[3] % cubes_size[2])

    n_cubes = cubes_per_x * cubes_per_y * cubes_per_z + additional_x_cube * cubes_per_y * cubes_per_z + \
              additional_y_cube * cubes_per_x * cubes_per_z + additional_z_cube * cubes_per_x * cubes_per_y

    if cubes_per_x * cubes_per_y * cubes_per_z == 0:
        print('Cubes are too big for img size')

    cubes = np.zeros((n_cubes, img.shape[2], cubes_size[0], cubes_size[1], cubes_size[2]))

    if bounding_box:
        cubes_bbox = []

    i = 0
    for n_cubes_x in range(cubes_per_x + additional_x_cube):
        for n_cubes_y in range(cubes_per_y + additional_y_cube):
            for n_cubes_z in range(cubes_per_z + additional_z_cube):

                if n_cubes_x == cubes_per_x: #additional cube
                    x_min = img.shape[0] - cubes_size[0]
                    x_max = img.shape[0]
                else:
                    x_min = n_cubes_x * cubes_size[0]
                    x_max = (n_cubes_x+1) * cubes_size[0]


                if n_cubes_y == cubes_per_y: #additional cube
                    y_min = img.shape[1] - cubes_size[1]
                    y_max = img.shape[1]
                else:
                    y_min = n_cubes_y * cubes_size[1]
                    y_max = (n_cubes_y+1) * cubes_size[1]


                if n_cubes_z == cubes_per_z: #additional cube
                    z_min = img.shape[3] - cubes_size[2]
                    z_max = img.shape[3]
                else:
                    z_min = n_cubes_z * cubes_size[2]
                    z_max = (n_cubes_z+1) * cubes_size[2]

                cubes[i,:,:,:,:] = np.transpose(img[x_min:x_max, y_min:y_max, :, z_min:z_max], (2,0,1,3))

                if bounding_box == True:
                    cubes_bbox.append(get_bboxes_coord(bboxes_coords, x_min, x_max, y_min, y_max, z_min, z_max))

                i += 1

    if cubes_bbox:
        return cubes, cubes_bbox
    else:
        return cubes

def save_cubes(out_data_dir_cubes, cubes, cubes_bbox=None, out_data_dir_bbox=None):


    for cube_number in range(cubes.shape[0]):
        # Normalize and rescale img and mask values
        img_cube = cubes[cube_number]

        if cubes_bbox is not None:
            bbox_coords = cubes_bbox[cube_number]
            np.save(out_data_dir_bbox+'/'+str(cube_number)+'_cube_bbox.npy', bbox_coords)

        img_cube = rescale_img_values(normalize_img_values(img_cube.astype(dtype=np.float64)))

        np.save(out_data_dir_cubes+'/'+str(cube_number)+'_cube.npy', img_cube)

def draw_cuboid(cube, pair):
    """

    :param cube: (channels,x,y,z)
    :param pair: (x1,y1,x2,y2,z1,z2)
    :return:
    """

    pair = pair.astype(np.int)
    if pair[0]<pair[2]:
        x_1 = pair[0]
        x_2 = pair[2]
    else:
        x_1 = pair[2]
        x_2 = pair[0]
        if x_1 == x_2:
            print('Invalid! x bbox coordinates for the 2 points are equal')
            return

    if pair[1]<pair[3]:
        y_1 = pair[1]
        y_2 = pair[3]
    else:
        y_1 = pair[3]
        y_2 = pair[1]
        if y_1 == y_2:
            print('Invalid! y bbox coordinates for the 2 points are equal')
            return
    if pair[4]<pair[5]:
        z_1 = pair[4]
        z_2 = pair[5]
    else:
        z_1 = pair[5]
        z_2 = pair[4]
        if z_1 == z_2:
            print('Invalid! z bbox coordinates for the 2 points are equal')
            return

    for z in range(z_1,z_2+1):
        cube[:, x_1:x_2,y_1,z] = 255
        cube[:, x_1:x_2, y_2,z] = 255

        cube[:, x_1,y_1:y_2,z] = 255
        cube[:, x_2, y_1:y_2, z] = 255


    return cube

def impose_bbox_cuboids(cube, bbox_coords):
    cube_imp = np.copy(cube)
    for pair in bbox_coords:
        if pair == []:
            return

        cube_imp = draw_cuboid(cube_imp,pair)

    return cube_imp

def create_nucleus_cuboids(coordinates_ng, margin,max_coor):

    bbox_coord = np.zeros(coordinates_ng.shape)
    for i, pair in enumerate(coordinates_ng):
        if pair == []:
            return
        if pair[0]<pair[3]:
            x_1 = pair[0]
            x_2 = pair[3]
        else:
            x_1 = pair[3]
            x_2 = pair[0]


        if pair[1]<pair[4]:
            y_1 = pair[1]
            y_2 = pair[4]
        else:
            y_1 = pair[4]
            y_2 = pair[1]

        if pair[2]<pair[5]:
            z_1 = pair[2]
            z_2 = pair[5]
        else:
            z_1 = pair[5]
            z_2 = pair[2]

        bbox_coord[i] = np.array([max(x_1-margin,0), max(y_1-margin,0), min(x_2+margin,max_coor[0]-1),
                                  min(y_2+margin,max_coor[1]-1), max(z_1-margin,0),min(z_2+margin,max_coor[2]-1)]).astype(np.int)

    return bbox_coord











