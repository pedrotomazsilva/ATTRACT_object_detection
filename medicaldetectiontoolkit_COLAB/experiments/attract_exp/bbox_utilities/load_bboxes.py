import numpy as np
from image_utilities.image_processing import impose_bbox_cuboids
from image_utilities.image_slicer import slice_and_select_cuboids
from bbox_utilities.create_manual_bboxes import read_image


def show_bboxes(image_path, bbox_path, rgb_order):
    bbox_coords = np.load(bbox_path, allow_pickle=True)
    bbox_coords = np.delete(bbox_coords, np.where(bbox_coords==None)[0],axis=0) #delete None rows
    bbox_coords = bbox_coords.astype(np.int16)
    image = read_image(image_path, rgb_order)
    image_w_bboxes = impose_bbox_cuboids(image,bbox_coords)
    slice_and_select_cuboids(image_w_bboxes, channels_first=True)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', required=True,
                        help='path to the czi image')
    parser.add_argument('--coordinates_file_path', required=True,
                        help='path of the coordinates file')

    parser.add_argument('--color_order', required=True,
                        help='order of the rgb colors in the czi image ex. gbr')

    args = parser.parse_args()
    show_bboxes(args.image_path, args.coordinates_file_path, args.color_order)