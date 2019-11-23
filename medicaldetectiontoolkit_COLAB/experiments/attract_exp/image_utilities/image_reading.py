import czifile
import numpy as np


def read_image(image_path, rgb_order):
    #return image shape (n_channels, x, y, z)
    image = czifile.imread(image_path)
    image = image.squeeze()
    image = image.transpose((0,2,3,1))
    uint16_max=65535

    image_r = ((np.expand_dims(image[rgb_order.index('r')],axis=0)/uint16_max)).astype(np.float32)
    image_g = ((np.expand_dims(image[rgb_order.index('g')],axis=0)/uint16_max)).astype(np.float32)
    image_b = np.zeros((1, image.shape[1], image.shape[2], image.shape[3])).astype(np.float32)
    image = np.concatenate((image_r,image_g,image_b),axis=0)

    return image