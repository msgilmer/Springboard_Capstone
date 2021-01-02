import keras.backend as K
import tensorflow as tf
from streamlit import cache

@cache
def maestro_loss_wr(harshness): 
    """A loss function which, in addition to penalizing for misclassification on the 
    first n_keys_piano (88) elements, includes a term proportional to the relative
    error in the prediction of the last element (which repesents the duration). 
    The proportionality constant is the 'harshness' of the maestro in regards to
    timing."""
    def maestro_loss(ytrue, ypred):
        # Standard binary cross-entropy
        bce_loss = - K.mean(ytrue[:, :-1] * K.log(ypred[:, :-1]) + (1 - ytrue[:, :-1]) * \
                     K.log(1 - ypred[:, :-1]))

        # Duration error term
        dur_loss = 2 * harshness * K.mean(K.abs((ytrue[:, -1] - ypred[:, -1]) / \
                                      (ytrue[:, -1] + ypred[:, -1] + K.epsilon())))
        
        if (dur_loss > bce_loss):   # Often times, ytrue[:, -1] elements will be zero
            return bce_loss * 2     # This may spike dur_loss. To control, I limit it
                                    # so that it never exceeds the bce_loss.
        return bce_loss + dur_loss
    
    return maestro_loss

@cache
def precision_mod(ytrue, ypred):
    """Just a modified precision excluding the last element (which is not a classification)"""

    true_positives = K.sum(K.round(ytrue[:, :-1] * ypred[:, :-1]))
    pred_positives = K.sum(K.round(ypred[:, :-1]))
    return true_positives / (pred_positives + K.epsilon())

@cache
def recall_mod(ytrue, ypred):
    """Just a modified recall excluding the last element (which is not a classification)"""

    true_positives = K.sum(K.round(ytrue[:, :-1] * ypred[:, :-1]))
    poss_positives = K.sum(ytrue[:, :-1])
    return true_positives / (poss_positives + K.epsilon())

@cache
def f1_score_mod(ytrue, ypred):
    """Just a modified f1_score excluding the last element (which is not a classification)"""

    precision = precision_mod(ytrue, ypred)
    recall = recall_mod(ytrue, ypred)   
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

@cache
def dur_error(ytrue, ypred):
    """A new metric that only gives information on the error in duration predictions"""
    
    return 2 * K.mean(K.abs((ytrue[:, -1] - ypred[:, -1]) / (ytrue[:, -1] + ypred[:, -1] + \
                                                         K.epsilon())))
@cache
def maestro_dur_loss_wr(harshness):
    """The second term of the maestro loss, based purely on error in duration predictions.
    To be used as a metric in order to decompose the loss components during analysis"""
    def maestro_dur_loss(ytrue, ypred):

        return 2 * harshness * K.mean(K.abs((ytrue[:, -1] - ypred[:, -1]) / \
                                      (ytrue[:, -1] + ypred[:, -1] + K.epsilon())))
    return maestro_dur_loss
