from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from image_utilities.image_processing import impose_bbox_cuboids

class Slicer(object):
    def __init__(self, ax, X, slice_cubes):
        self.ax = ax
        self.ax.set_title('Click to change image and scroll to view depths\n\nCube number 0')
        self.slice_cubes = slice_cubes


        if self.slice_cubes:
            self.all_cubes = X
            self.cube_no = 0
            self.total_n_cubes = self.all_cubes.shape[0]
            self.X = self.all_cubes[self.cube_no]
        else:
            self.X = X

        self.slices = self.X.shape[3]
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def onclick(self, event):
        self.cube_no += 1
        self.ax.set_title('Cube number %d' % (self.cube_no))
        if self.cube_no < self.total_n_cubes:
            self.X = self.all_cubes[self.cube_no]
            self.update()


    def update(self):

        self.im.set_data(self.X[:, :, :,self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()





class CubesSlicer(object):
    def __init__(self, ax, cubes, cubes_mask, image_cubes_dict=None, images_per_figure=4): #cubes shape MASK OR IMAGE: (n_cubes,x,y,3,z)

        self.ax = ax
        self.image_cubes_dict = image_cubes_dict
        self.images_per_figure = images_per_figure
        self.all_cubes = cubes
        self.cubes = cubes[0:images_per_figure]
        self.all_cubes_mask = cubes_mask
        self.cubes_mask = cubes_mask[0:images_per_figure]
        self.im = [[] for i in range(2)]
        self.slices = cubes.shape[4]
        self.ind = self.slices//2
        self.showed_cubes = 0

        self.set_titles()
        self.showed_cubes = images_per_figure
        self.show_images()
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def set_titles(self):
        for n_cubes in range(self.cubes.shape[0]):
            if self.image_cubes_dict is not None:
                i = self.showed_cubes + n_cubes
                if i >= self.image_cubes_dict['P6'][0] and i <= self.image_cubes_dict['P6'][1]:
                    image_name = 'P6'
                elif i >= self.image_cubes_dict['P6_SF'][0] and i <= self.image_cubes_dict['P6_SF'][1]:
                    image_name = 'P6_SF'
                elif i >= self.image_cubes_dict['P6_AVM'][0] and i <= self.image_cubes_dict['P6_AVM'][1]:
                    image_name = 'P6_AVM'
                self.ax[0,n_cubes].set_title('Image %s' % image_name)

            self.ax[0,n_cubes].set_yticklabels([])
            self.ax[0,n_cubes].set_xticklabels([])
            self.ax[1, n_cubes].set_yticklabels([])
            self.ax[1, n_cubes].set_xticklabels([])

    def show_images(self):
        for i in range(self.cubes.shape[0]):
            self.im[0].append(self.ax[0,i].imshow(self.cubes[i, :, :, :, self.ind]))
            self.im[1].append(self.ax[1,i].imshow(self.cubes_mask[i, :, :, :, self.ind]))

    def onclick(self,event):
        self.cubes = self.all_cubes[self.showed_cubes:(self.showed_cubes+self.images_per_figure)]
        self.cubes_mask = self.all_cubes_mask[self.showed_cubes:(self.showed_cubes+self.images_per_figure)]
        self.showed_cubes += self.images_per_figure
        self.set_titles()
        self.update()

    def update(self):

        for i in range(self.cubes.shape[0]):
            self.im[1][i].set_data(self.cubes_mask[i, :, :, :, self.ind])
            self.im[0][i].set_data(self.cubes[i, :, :, :, self.ind])


        for i in range(self.cubes.shape[0]):
            self.im[0][i].axes.figure.canvas.draw()
            self.im[1][i].axes.figure.canvas.draw()

        self.ax[0,0].set_ylabel('slice %s' % self.ind)
        self.ax[1,0].set_ylabel('slice %s' % self.ind)




def slicer(image, slice_cubes=False, channels_first=False):

    if channels_first:
        if slice_cubes:
            image = np.transpose(image, (0,2,3,1,4))
        else:
            image = np.transpose(image, (1,2,0,3))

    fig, ax = plt.subplots(1, 1)
    tracker = Slicer(ax, image, slice_cubes)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    if slice_cubes:
        fig.canvas.mpl_connect('button_press_event', tracker.onclick)
    plt.show()



def slice_cubes(cubes,cubes_mask,images_per_figure = 4):

    cubes = np.transpose(cubes, (0,2,3,1,4))
    cubes_mask = np.transpose(cubes_mask, (0,2,3,1,4))

    fig, axes = plt.subplots(2, images_per_figure)
    fig.subplots_adjust(hspace=0,wspace=0)

    slicer = CubesSlicer(axes, cubes, cubes_mask,None,images_per_figure)
    fig.canvas.mpl_connect('scroll_event', slicer.onscroll)
    fig.canvas.mpl_connect('button_press_event', slicer.onclick)
    plt.show()


def order_dirs_imgs(element):
    return int(element.split('_')[0])

def order_dirs_masks(element):
    return int(element.split('_')[1])



'''
----------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------
'''





class IndexTracker(object):
    def __init__(self, ax, X, channels_first):
        self.ax = ax
        self.X = X
        self.previous_X = np.copy(X)
        if channels_first:
            self.X = np.transpose(self.X, (1,2,0,3))
        self.slices = X.shape[3]
        self.ind = self.slices//2
        self.draw = False

        self.im = ax.imshow(self.X[:, :, :, self.ind])
        self.n_points=0
        self.bboxes_coords = None
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def onpresskey(self, event):
        if event.key == 'd':
            self.draw = True
            print('Drawing cuboids')
        if event.key == 'c':
            self.draw = False
            print('Drawing cuboids deactivated')
        if event.key == 'x':
            self.X = self.previous_X
            self.bboxes_coords = np.delete(self.bboxes_coords, -1,0)

    def onclick(self, event):
        # Create a Rectangle patch
        if self.draw:
            self.n_points += 1
            if self.n_points == 2:

                self.point2_x = event.xdata
                self.point2_y = event.ydata
                self.point2_z = self.ind

                bbox_coords = np.expand_dims(np.array([self.point1_y, self.point1_x, self.point2_y,
                                            self.point2_x, self.point1_z, self.point2_z]),axis=0)

                self.previous_X = np.copy(self.X)
                self.X = np.transpose(impose_bbox_cuboids(
                    np.transpose(self.X,(2,0,1,3)), bbox_coords=bbox_coords), (1,2,0,3))
                self.update()

            else:
                self.point1_x = event.xdata
                self.point1_y = event.ydata
                self.point1_z = self.ind




    def update(self):

        if self.n_points == 2:


            bbox_array = np.array([[self.point1_y, self.point1_x, self.point2_y,
                                    self.point2_x, self.point1_z, self.point2_z]])

            if self.bboxes_coords is None:
                self.bboxes_coords = bbox_array
            else:
                self.bboxes_coords = np.concatenate((self.bboxes_coords, bbox_array),axis=0)

            self.n_points=0


        self.im.set_data(self.X[:, :, :,self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()




def slice_and_select_cuboids(image, channels_first=False):

    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, image, channels_first=channels_first)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    fig.canvas.mpl_connect('button_press_event', tracker.onclick)
    fig.canvas.mpl_connect('key_press_event', tracker.onpresskey)
    plt.show()

    return tracker.bboxes_coords
