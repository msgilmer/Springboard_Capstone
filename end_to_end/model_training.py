import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.callbacks import ModelCheckpoint, TerminateOnNaN
from keras.optimizers import RMSprop

import keras.backend as K
import tensorflow as tf

from decimal import Decimal

def lstm(n_lstm_layers = 2, n_dense_layers = 1, n_lstm_nodes = 512, \
         dropout_rate = 0.4, n_keys_piano = 88, window_size = 16, \
         n_dur_nodes = 20, max_norm_value = None):
    """Generate a keras Sequential model of the form as described in Figure 16
    of https://www.tandfonline.com/doi/full/10.1080/25765299.2019.1649972"""
    if (max_norm_value):
        kernel_constraint = tf.keras.constraints.max_norm(max_norm_value)
    else:
        kernel_constraint = None
    model = Sequential()
    model.add(LSTM(n_lstm_nodes, return_sequences = True, input_shape = \
                   (window_size, n_keys_piano + n_dur_nodes,)))
    model.add(Dropout(dropout_rate))
    for i in range(1, n_lstm_layers - 1):
        model.add(LSTM(n_lstm_nodes, return_sequences = True, kernel_constraint = \
                                                            kernel_constraint))
        model.add(Dropout(dropout_rate))
    model.add(LSTM(n_lstm_nodes, kernel_constraint = kernel_constraint))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_lstm_nodes // 2, kernel_constraint = kernel_constraint))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    for i in range(n_dense_layers - 1):
        model.add(Dense(n_lstm_nodes // 2, kernel_constraint = \
                                             kernel_constraint))
        model.add(Dropout(dropout_rate))
    model.add(Dense(n_keys_piano + n_dur_nodes, kernel_constraint = \
                                                  kernel_constraint))
    model.add(Activation('sigmoid'))
    return model

def maestro_loss_wr(harshness, n_dur_nodes): 
    """A loss function which, in addition to penalizing for misclassification on
    the first n_keys_piano elements, includes a term proportional to the
    relative error in the prediction of the last n_dur_nodes elements (whose
    mean represents the duration). The proportionality constant is the 'harshness'
    of the maestro in regards to timing."""
    def maestro_loss(ytrue, ypred):
        # Standard binary cross-entropy
        bce_loss = - K.mean(ytrue[:, :-n_dur_nodes] * K.log(ypred[:, \
                            :-n_dur_nodes]) + (1 - ytrue[:, :-n_dur_nodes]) * \
                            K.log(1 - ypred[:, :-n_dur_nodes]))

        # Duration error term
        dur_loss = 2 * harshness * K.mean(K.abs(K.mean(ytrue[:, -n_dur_nodes:], \
                       axis = 1) - K.mean(ypred[:, -n_dur_nodes:], axis = 1)) / \
                       (K.mean(ytrue[:, -n_dur_nodes:], axis = 1) + \
                       K.mean(ypred[:, -n_dur_nodes:], axis = 1) + K.epsilon()))
        
        if (dur_loss > bce_loss):   # Often times, ytrue[:, :-n_dur_nodes]
            return bce_loss * 2     # elements will be zero (for a rest). This 
                                    # may spike dur_loss.  To control, I limit
                                    # it so that it never exceeds the bce_loss.
        return bce_loss + dur_loss
    
    return maestro_loss
def precision_mod_wr(n_dur_nodes):
    def precision_mod(ytrue, ypred):
        """Just a modified precision excluding the last n_dur_nodes elements
        (which are not classification nodes)"""

        true_positives = K.sum(K.round(ytrue[:, :-n_dur_nodes] * \
                                       ypred[:, :-n_dur_nodes]))
        pred_positives = K.sum(K.round(ypred[:, :-n_dur_nodes]))
        return true_positives / (pred_positives + K.epsilon())
    return precision_mod

def recall_mod_wr(n_dur_nodes):
    def recall_mod(ytrue, ypred):
        """Just a modified recall excluding the last n_dur_nodes elements (which
        are not classification nodes)"""

        true_positives = K.sum(K.round(ytrue[:, :-n_dur_nodes] * \
                                       ypred[:, :-n_dur_nodes]))
        poss_positives = K.sum(ytrue[:, :-n_dur_nodes])
        return true_positives / (poss_positives + K.epsilon())
    return recall_mod

def f1_score_mod_wr(n_dur_nodes):
    def f1_score_mod(ytrue, ypred):
        """Just a modified f1_score excluding the last n_dur_nodes elements
        (which are not classification nodes)"""

        precision = precision_mod_wr(n_dur_nodes)(ytrue, ypred)
        recall = recall_mod_wr(n_dur_nodes)(ytrue, ypred)   
        return 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_score_mod

# Test this versus the one below
def dur_error_wr(n_dur_nodes):
    def dur_error(ytrue, ypred):
        """A new metric that only gives information on the error in duration
        predictions"""
    
        return 2 * K.mean(K.abs((K.mean(ytrue[:, -n_dur_nodes:], axis = 1) - \
                   K.mean(ypred[:, -n_dur_nodes:], axis = 1)) / \
                   (K.mean(ytrue[:, -n_dur_nodes:], axis = 1) + \
                   K.mean(ypred[:, -n_dur_nodes:], axis = 1) + K.epsilon())))
    return dur_error

def maestro_dur_loss_wr(harshness, n_dur_nodes):
    """The second term of the maestro loss, based purely on error in duration
    predictions. To be used as a metric in order to decompose the loss
    components during analysis"""
    def maestro_dur_loss(ytrue, ypred):

        return 2 * harshness * K.mean(K.abs((K.mean(ytrue[:, -n_dur_nodes:], \
                   axis = 1) - K.mean(ypred[:, -n_dur_nodes:], axis = 1)) / \
                  (K.mean(ytrue[:, -n_dur_nodes:], axis = 1) + \
                  K.mean(ypred[:, -n_dur_nodes:], axis = 1) + K.epsilon())))
    return maestro_dur_loss

def get_optimizer(lr, clipnorm, clipvalue):
    """Return the RMSprop optimizer with the above parameters"""

    if (lr or clipnorm or clipvalue):
        if (lr):          # It's required that the first argument to RMSprop is
                          # not None
            return RMSprop(lr = lr, clipnorm = clipnorm, clipvalue = clipvalue)
        elif (clipnorm):
            return RMSprop(clipnorm = clipnorm, clipvalue = clipvalue)
        else: # clipvalue
            return RMSprop(clipvalue = clipvalue)
    return RMSprop()   # TypeError when all are None, so do this instead

def get_filename(n_dur_nodes, n_lstm_layers, n_dense_layers, n_lstm_nodes, \
                 dropout_rate, base_nm = 'best_maestro_model_ext_e2e', \
                 max_norm_value = None, lr = None, clipnorm = None,
                 clipvalue = None):
    """Get the filename to save the best model to based on the passed
    parameters."""

    filename = 'best_maestro_model_ext{0}_e2e_{1}_{2}_{3}_{4}'\
               .format(n_dur_nodes, n_lstm_layers, n_dense_layers, \
               n_lstm_nodes, str(dropout_rate).replace('.', 'pt'))
    if (max_norm_value):
        filename += '_mnv_{}'.format(Decimal(max_norm_value))
    if (lr):
        filename += '_lr_{}'.format('%.0e' % Decimal(lr))
    if (clipnorm):
        filename += '_cn_{}'.format(str(clipnorm).replace('.', 'pt'))     
    if (clipvalue):
        filename += '_cv_{}'.format(str(clipvalue).replace('.', 'pt'))

    return filename

def generate_cols_dict(history):
    """Return a mapping of desired column names to the corresponding columns in
    the history dictionary (previously history.history where history is the
    return value of model.train)"""
    return {'maestro_loss': history['loss'], 'f1_score': \
            history['f1_score_mod'], 'precision': history['precision_mod'], \
            'recall': history['recall_mod'], 'dur_error': history['dur_error'],\
            'dur_loss': history['maestro_dur_loss'], 'val_maestro_loss': \
            history['val_loss'], 'val_f1_score': history['val_f1_score_mod'], \
            'val_precision': history['val_precision_mod'], \
            'val_recall': history['val_recall_mod'], 'val_dur_error': \
            history['val_dur_error'], 'val_dur_loss': \
            history['val_maestro_dur_loss']}

def save_performance_data(history, filename, path = \
                          '../model_data/performance_data/'):
    """Save the performance data stored in the history dictionary (previously
    history.history where history is the return value of model.train) to a csv
    file."""
    
    # In all preliminary tests model training has failed at some point when the
    # loss becomes NaN
    if (len(history['val_loss']) < len(history['loss'])):
        # a NaN during training
        for key, value in history.items():
            # pd.DataFrame requires value lengths to be equal:
            if (key[:3] == 'val'):      
                value.append(np.nan)
                
    df = pd.DataFrame(generate_cols_dict(history))
    df.index.name = 'Epochs'
    df.to_csv(path + filename + '.csv')


def train_lstm_model(X_train, X_val, y_train, y_val, n_dur_nodes = 20, \
                     n_lstm_layers = 2, n_dense_layers = 1, n_lstm_nodes = 512,\
                     dropout_rate = 0.4, batch_size = 512, harshness = 0.05, \
                     lr = None, clipnorm = None, clipvalue = None, \
                     max_norm_value = None, epochs = 150):
    """Train a model using the passed parameters, the data, and using the
    RMSprop optimizer. Write the best model as a .h5 and a .csv containing
    columns for the training and validation custom loss and
    metrics. Returns nothing."""

    model = lstm(n_lstm_layers = n_lstm_layers, n_dense_layers = \
                 n_dense_layers, n_lstm_nodes = n_lstm_nodes, dropout_rate = \
                 dropout_rate, max_norm_value = max_norm_value)

    opt = get_optimizer(lr, clipnorm, clipvalue)
        
    model.compile(loss = maestro_loss_wr(harshness, n_dur_nodes), optimizer = \
                  opt, metrics = [f1_score_mod_wr(n_dur_nodes), \
                  recall_mod_wr(n_dur_nodes), precision_mod_wr(n_dur_nodes), \
                  dur_error_wr(n_dur_nodes), maestro_dur_loss_wr(harshness, \
                                                                 n_dur_nodes)])

    filename = get_filename(n_dur_nodes, n_lstm_layers, n_dense_layers, \
                            n_lstm_nodes, dropout_rate, \
                            max_norm_value = max_norm_value, lr = lr, \
                            clipnorm = clipnorm, clipvalue = clipvalue)
                                   
    mc = ModelCheckpoint('../models/' + filename + '.h5', monitor = 'val_loss',\
                         mode = 'min', save_best_only = True, verbose = 1)
                                   
    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = \
                        epochs, validation_data = (X_val, y_val), verbose = 2, \
                        callbacks = [mc, TerminateOnNaN()])

    save_performance_data(history.history, filename)
