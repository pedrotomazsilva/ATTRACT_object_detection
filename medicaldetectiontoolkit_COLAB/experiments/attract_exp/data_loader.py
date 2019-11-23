import numpy as np
import os

def get_train_generators(cf, logger):

    batchgen_train = DataLoader(partition='train', configs=cf)
    batchgen_val = DataLoader(partition='val', configs=cf, shuffle=False)


    return {'train': batchgen_train, 'val_sampling': batchgen_val, 'n_val': len(batchgen_val)}

def get_test_generator(cf, logger):

    batchgen = DataLoader(partition='val', configs=cf, shuffle=False)


    return {'test': batchgen, 'n_test': len(batchgen)}


class DataLoader:
    #BOUNDING BOXES JUST WORK FOR BATCHSIZE=1 FOR NOW because of the number of rois
    def __init__(self, partition, configs, shuffle=True): #pid: batch_id


        self.partition = partition
        self.root_dir = configs.root_dir
        self.list_IDs = sorted(os.listdir(self.root_dir + '/experiments/attract_exp/data/' + partition
                                          + '/img/'), key=self.order_dirs)

        self.dim = configs.img_shape
        self.batch_size = configs.batch_size
        self.shuffle = shuffle
        self.generate_bboxes = configs.generate_bboxes
        self.on_epoch_end()

        # data is now stored in self._data.

    def __iter__(self):
        return self

    def __next__(self):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[self.index * self.batch_size:(self.index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Update index
        self.index += 1

        # Generate data
        if self.generate_bboxes:
            images, bb_target = self.__data_generation(list_IDs_temp)
        else:
            images = self.__data_generation(list_IDs_temp)

        self.pid += 1

        if self.partition == 'train':
            if bb_target[0] is None:
                bb_target = np.array([[]])
                roi_labels = np.array([[]])
                roi_masks = np.array([[]])
            else:
                roi_labels = np.zeros((self.batch_size, bb_target.shape[1]))
                roi_masks = np.zeros((self.batch_size, bb_target.shape[1], self.dim[1], self.dim[2], self.dim[3],1))

            return {'data': images, 'pid': self.pid,
                    'bb_target': bb_target,
                    #'roi_labels': np.zeros((self.batch_size, bb_target.shape[1], 1)),
                    'roi_labels': roi_labels,
                    'roi_masks': roi_masks,
                    'seg': np.zeros(
                        (self.batch_size, 1, self.dim[1], self.dim[2], self.dim[3]))}
        if self.partition == 'val':
            if bb_target[0] is None:
                bb_target = np.array([[]])
                roi_labels = np.array([[]])
                roi_masks = np.array([[]])
            else:
                roi_labels = np.zeros((self.batch_size, bb_target.shape[1]))
                roi_masks = np.zeros((self.batch_size, bb_target.shape[1], self.dim[1], self.dim[2], self.dim[3], 1))

            return {'data': images, 'pid': self.pid,
                    'bb_target': bb_target,
                    'roi_labels': roi_labels,
                    'roi_masks': roi_masks,
                    'seg': np.zeros(
                        (self.batch_size, 1, self.dim[1], self.dim[2], self.dim[3]))}

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.index = 0
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Initialization
        images = np.empty((self.batch_size, *self.dim))

        if list_IDs_temp == []:
            print('Warning: all batch elements were used, re initialize the generator')

        # Generate data
        for i, ID_path in enumerate(list_IDs_temp):

            self.pid = int(ID_path.split('_')[0])
            images[i, ] = np.transpose((np.load(self.root_dir + '/experiments/attract_exp/data/' + self.partition +'/img/'+ ID_path)),
                                       (0,1,2,3)).astype(np.float)
            if self.generate_bboxes:
                bboxes_coords = np.expand_dims(np.array(
                    np.load(self.root_dir + '/experiments/attract_exp/data/'+self.partition +
                            '/bboxes/' + ID_path[0:-4] + '_bbox.npy')),axis=0)



        if self.generate_bboxes:
            return images, bboxes_coords
        else:
            return images

    def order_dirs(self, element):
        return int(element.split('_')[0])


