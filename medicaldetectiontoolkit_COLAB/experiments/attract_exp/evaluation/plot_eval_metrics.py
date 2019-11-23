import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def read_metrics(fold_path): #Currently class_loss just works for 1 class but loss works for any number of classes

    metrics_path = os.path.join(fold_path,'last_checkpoint','monitor_metrics.pickle')
    metrics_dict = pd.read_pickle(metrics_path)

    loss_dict = {
        'train':{
            'loss':[],
            'class_loss':[]
        },
        'val':{
            'loss':[],
            'class_loss':[]
        }
    }

    for phase in metrics_dict:
        for epoch_values in metrics_dict[phase]['monitor_values']:
            loss_vec = []
            class_loss_vec = []
            if len(epoch_values)==0:
                continue
            for batch in epoch_values:
                loss_vec.append(batch['loss'])
                class_loss_vec.append(batch['class_loss'])
            loss_dict[phase]['loss'].append(np.mean(loss_vec))
            loss_dict[phase]['class_loss'].append(np.mean(class_loss_vec))

    return loss_dict

def plot_training_evolution(loss_dict):

    plt.xlabel('Epoch number')
    plt.ylabel('Loss')
    epochs_vec = np.arange(len(loss_dict['train']['loss']))
    plt.plot(epochs_vec, loss_dict['train']['loss'], '^b')
    plt.plot(epochs_vec, loss_dict['val']['loss'], 'xr')
    plt.legend(['train', 'validation'],
               loc='upper left')
    plt.show()
    return

fold_dir = 'C:/Users/pedro/Desktop/medicaldetectiontoolkit_COLAB/experiments/attract_exp/fold_0'
loss_dict=read_metrics(fold_dir)
plot_training_evolution(loss_dict)
