import data_read_and_process as drap
import numpy as np

def test_note_to_piano_idx():

    correct_answers = {'A0': 0, 'A#0': 1, 'B-0': 1, 'B0': 2, 'C1': 3, 'C#1': 4,\
                       'D-1': 4, 'D1': 5, 'D#1': 6, 'E-1': 6, 'E1': 7, 'F1': 8,\
                       'F#1': 9, 'G-1': 9, 'G1': 10, 'G#1': 11, 'A-1': 11,\
                       'A1': 12, 'A#1': 13, 'B-1': 13, 'B1': 14, 'C2': 15,\
                       'C#2': 16, 'D-2': 16, 'D2': 17, 'D#2': 18, 'E-2': 18,\
                       'E2': 19, 'F2': 20, 'F#2': 21, 'G-2': 21, 'G2': 22,\
                       'G#2': 23, 'A-2': 23, 'A2': 24, 'A#2': 25, 'B-2': 25,\
                       'B2': 26, 'C3': 27, 'C#3': 28, 'D-3': 28, 'D3': 29,\
                       'D#3': 30, 'E-3': 30, 'E3': 31, 'F3': 32, 'F#3': 33,\
                       'G-3': 33, 'G3': 34, 'G#3': 35, 'A-3': 35, 'A3': 36,\
                       'A#3': 37, 'B-3': 37, 'B3': 38, 'C4': 39, 'C#4': 40,\
                       'D-4': 40, 'D4': 41, 'D#4': 42, 'E-4': 42, 'E4': 43,\
                       'F4': 44, 'F#4': 45, 'G-4': 45, 'G4': 46, 'G#4': 47,\
                       'A-4': 47, 'A4': 48, 'A#4': 49, 'B-4': 49, 'B4': 50,\
                       'C5': 51, 'C#5': 52, 'D-5': 52, 'D5': 53, 'D#5': 54,\
                       'E-5': 54, 'E5': 55, 'F5': 56, 'F#5': 57, 'G-5': 57,\
                       'G5': 58, 'G#5': 59, 'A-5': 59, 'A5': 60, 'A#5': 61,\
                       'B-5': 61, 'B5': 62, 'C6': 63, 'C#6': 64, 'D-6': 64,\
                       'D6': 65, 'D#6': 66, 'E-6': 66, 'E6': 67, 'F6': 68,\
                       'F#6': 69, 'G-6': 69, 'G6': 70, 'G#6': 71, 'A-6': 71,\
                       'A6': 72, 'A#6': 73, 'B-6': 73, 'B6': 74, 'C7': 75,\
                       'C#7': 76, 'D-7': 76, 'D7': 77, 'D#7': 78, 'E-7': 78,\
                       'E7': 79, 'F7': 80, 'F#7': 81, 'G-7': 81, 'G7': 82,\
                       'G#7': 83, 'A-7': 83, 'A7': 84, 'A#7': 85, 'B-7': 85,\
                       'B7': 86, 'C8': 87}

            
    for letter in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
        for addon in ['', '-', '#']:
            for octave in range(9):
                composite = letter + addon + str(octave)
                if (composite in correct_answers):
                    assert(drap.note_to_piano_idx(composite) == \
                           correct_answers[composite])
                else:
                    assert(drap.note_to_piano_idx(composite) == -1)

    for letter in ['H', '5', ',', 'c']:
        for addon in ['', '-', '#']:
            for octave in range(9):
                composite = letter + addon + str(octave)
                assert(drap.note_to_piano_idx(composite) == -1)

    for letter in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
        for addon in ['+', 'b', ',', 'B', '8']:
            for octave in range(9):
                composite = letter + addon + str(octave)
                assert(drap.note_to_piano_idx(composite) == -1)

    for letter in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
        for addon in ['', '-', '#']:
            for octave in ['9', 'c', 'C', ',', '']:
                composite = letter + addon + octave
                assert(drap.note_to_piano_idx(composite) == -1)

def test_transpose_sequence(n_keys_piano = 88):

    n_tests = 5
    n_sequences = 5
    for i in range(n_tests):
        transposition = np.random.randint(0, 12) # random amount to transpose by
        test_sequence = []
        correct_result = []
        for j in range(n_sequences):
            random_size = np.random.randint(0, n_keys_piano)
            random_indices = np.random.choice(n_keys_piano, \
                                            size = random_size, replace = False)
            vector = np.zeros(n_keys_piano)
            np.put(vector, random_indices, 1)
            test_sequence.append(vector)
            transposed_indices = []
            for index in random_indices:
                transposed_index = index + transposition
                if (transposed_index >= n_keys_piano):
                    transposed_index -= n_keys_piano
                transposed_indices.append(transposed_index)
            vector = np.zeros(n_keys_piano)
            np.put(vector, transposed_indices, 1)
            correct_result.append(vector)

        result = drap.transpose_sequence(test_sequence, transposition)
        for j in range(n_sequences):
            assert((result[j] == correct_result[j]).all())
    
        
    
