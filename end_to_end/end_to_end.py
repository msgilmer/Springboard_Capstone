import data_read_and_process as drap
import model_training as mt

from os import listdir, path
import sys

import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    
    directory = '../chopin'
    keys = []
    songs = []
    for filename in listdir(directory):
        if (filename == 'chpn_op35_2.mid'):  # skip this file,
                                             # it has two separate parts.
            continue
        f = path.join(directory, filename)
        result = drap.read_midi(f)
        if (result == -1):                   # If any file read is unsuccessful,
                                             # abort
            sys.exit('Failed to read file {}'.format(filename))
        keys.append(result[0])
        songs.append(result[1])

    keys_by_song, notes_by_song = drap.split_songs(keys, songs)

    chopin_sequences = drap.songs_to_sequences(notes_by_song)

    # Often times, songs finish with a rest, remove these
    for i in range(len(chopin_sequences)):
        while(sum(chopin_sequences[i][-1][:-1]) == 0):  
            chopin_sequences[i] = chopin_sequences[i][:-1]

    transposed_chopin_sequences = drap.transpose_sequences(chopin_sequences, \
                                                           keys_by_song)

    n_dur_nodes = 20
    extended_transposed_chopin_sequences = []
    for sequence in transposed_chopin_sequences:
        sequence = np.concatenate([sequence, np.tile(sequence[:, -1], \
                        (n_dur_nodes - 1, 1)). transpose()], axis = 1)
        extended_transposed_chopin_sequences.append(sequence)

    # Apply window function of default size 16
    X, y = drap.sequences_to_inputs(extended_transposed_chopin_sequences)
                           
    X, y = shuffle(X, y, random_state = 42)

    maximum_duration = max(X[:, :, -1].max(), y[:, -1].max())
    print('Maximum Duration = {}'.format(maximum_duration))
    # Keep variable so one can multiply the durations by this after music
    # generation to convert back into seconds
    X[:, :, -1] /= maximum_duration
    y[:, -1] /= maximum_duration

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.3, \
                                                      random_state = 42)

    mt.train_lstm_model(X_train, X_val, y_train, y_val)
