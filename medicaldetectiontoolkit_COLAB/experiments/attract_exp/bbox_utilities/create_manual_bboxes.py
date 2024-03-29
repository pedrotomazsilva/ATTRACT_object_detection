from image_utilities.image_slicer import slice_and_select_cuboids
import pandas as pd
import numpy as np
import czifile


def read_image(image_path, rgb_order):
    #return image shape (n_channels, x, y, z)
    image = czifile.imread(image_path)
    image = image.squeeze()
    image = image.transpose((0,2,3,1))
    uint16_max=65535

    image_r = ((np.expand_dims(image[rgb_order.index('r')],axis=0)/uint16_max)*255).astype(np.int16)
    image_g = ((np.expand_dims(image[rgb_order.index('g')],axis=0)/uint16_max)*255).astype(np.int16)
    image_b = np.zeros((1, image.shape[1], image.shape[2], image.shape[3])).astype(np.int16)
    image = np.concatenate((image_r,image_g,image_b),axis=0)

    return image

def create_bounding_boxes(image_path, bounding_box_coordinates_path, rgb_order):
    image = read_image(image_path, rgb_order)

    #D - activates draw - just click in 2 points and it will save and plot the cuboid. Be careful not to click on anything
    # but the points you desire while you are with the drawing activated
    #A - deactivates draw
    #X - removes previous cuboid-but only the most recent one, it cannot elimate other than the current one even i you keep
    #pressing
    #The results are saved in the shape (y1,x1,y2,x2,z1,z2)
    #You can use zoom in and zoom out (be careful not to do it while on drawing mode)
    #The boxes may sometimes appear incomplete. Don't worry, if you zoom in you will be able to see all the lines.

    points = slice_and_select_cuboids(image, channels_first=True)
    #points shape (y1,x1,y2,x2,z1,z2)
    np.save(bounding_box_coordinates_path+'.npy', points)
    df = pd.DataFrame(points)
    df.to_excel(bounding_box_coordinates_path+'.xlsx', index=False, header=['y1','x1','y2','x2','z1','z2'])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True,
                        help='path to the czi image')
    parser.add_argument('--coordinates_file_name', required=True,
                        help='desired name for the coordinates file')

    parser.add_argument('--color_order', required=True,
                        help='order of the rgb colors in the czi image ex. gbr')

    args = parser.parse_args()
    create_bounding_boxes(args.image_path, args.coordinates_file_name, args.color_order)