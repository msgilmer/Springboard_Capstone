import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

import matplotlib.pyplot as plt
from matplotlib import rcParams

def lstm(n_lstm_layers = 2, n_dense_layers = 1, n_lstm_nodes = 512, \
         dropout_rate = 0.4, n_keys_piano = 88, window_size = 16, \
         n_dur_nodes = 20):
    """Generate a keras Sequential model of the form as described in Figure 16
    of https://www.tandfonline.com/doi/full/10.1080/25765299.2019.1649972"""
    
    model = Sequential()
    model.add(LSTM(n_lstm_nodes, return_sequences = True, input_shape = \
                   (window_size, n_keys_piano + n_dur_nodes,)))
    model.add(Dropout(dropout_rate))
    for i in range(1, n_lstm_layers - 1):
        model.add(LSTM(n_lstm_nodes, return_sequences = True))
        model.add(Dropout(dropout_rate))
    model.add(LSTM(n_lstm_nodes))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_lstm_nodes // 2))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    for i in range(n_dense_layers - 1):
        model.add(Dense(n_lstm_nodes // 2))
        model.add(Dropout(dropout_rate))
    model.add(Dense(n_keys_piano + n_dur_nodes))
    model.add(Activation('sigmoid'))
    return model

epsilon = 1.e-7
def precision_mod_thresh(ytrue, ypred, n_dur_nodes = 20, thresh = 0.5):
    """Just a modified precision excluding the last n_dur_nodes elements (which
    are not classification nodes) and with probability threshold parameter:
    thresh."""

    true_positives = np.sum(np.where(ytrue[:, :-n_dur_nodes] * \
                                     ypred[:, :-n_dur_nodes] >= thresh, 1, 0))
    pred_positives = np.sum(np.where(ypred[:, :-n_dur_nodes] >= thresh, 1, 0))
    return true_positives / (pred_positives + epsilon)

def recall_mod_thresh(ytrue, ypred, n_dur_nodes = 20, thresh = 0.5):
    """Just a modified recall excluding the last n_dur_nodes elements (which
    are not classification nodes) and with probability threshold parameter:
    thresh."""

    true_positives = np.sum(np.where(ytrue[:, :-n_dur_nodes] * \
                                     ypred[:, :-n_dur_nodes] >= thresh, 1, 0))
    poss_positives = np.sum(ytrue[:, :-n_dur_nodes])
    return true_positives / (poss_positives + epsilon)

if __name__ == '__main__':
    model = lstm(n_lstm_nodes = 1024)
    model.load_weights('../models/best_maestro_model_weights_ext20_2_1_1024' +\
                       '_0pt4_mnv_2.h5')
    X_val = np.load('../train_and_val/X_val_ext.npy')
    y_val = np.load('../train_and_val/y_val_ext.npy')
    batch_size = 512  # why is this needed for predict method?
    pred = model.predict(X_val, batch_size = batch_size)
    n_steps = 1000
    precisions = []
    recalls = []
    f_measures = []
    for i in range(1, n_steps):
        thresh = i / n_steps
        precision = precision_mod_thresh(y_val, pred, thresh = thresh)
        recall = recall_mod_thresh(y_val, pred, thresh = thresh)
        precisions.append(precision)
        recalls.append(recall)
        f_measures.append(2 * (precision * recall) / (precision + recall + \

                                                      epsilon))
    rcParams.update({'font.size': 22})       
    x = np.arange(1, n_steps) / n_steps
    plt.figure(figsize = (12, 8))
    plt.plot(x, np.array(precisions) * 100, label = 'precision')
    plt.plot(x, np.array(recalls) * 100, label = 'recall')
    plt.plot(x, np.array(f_measures) * 100, label = 'f-measure')
    plt.legend()
    plt.xlabel('Threshold')
    plt.ylabel('Percent')
    plt.xlim(0, 1)
    plt.ylim(0, 100)
    plt.title('Threshold Choice Guide')
    plt.savefig('../images/precision_and_recall.jpg')
    
                       
