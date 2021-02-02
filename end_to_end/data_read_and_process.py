# Load functions from music21 (https://web.mit.edu/music21/doc/)to be used for
# parsing MIDI files
from music21 import converter, instrument, note, chord, key, tempo, duration

import numpy as np


def read_midi(file, time_tol = 1.e-3):
    """This function reads a midi file for the notes, offsets, and durations
    for the piano relevant piano part. The data needs to be further processed
    because there are many notes occurring at identical offsets (meaning they
    start at the same time) but have different durations. For a more sequential
    representation, I convert this data into a sequence in which there is no
    overlap, i.e. any time there is a change in the state of the piano keyboard,
    a new element is created describing this state along with a duration for
    that state. This format will be much easier to use for encoding the data
    into the LSTM network input later on. These sequences, are  returned along
    with the musical keys (keys instead of key because there are key changes) of
    the song (we need this because we will eventually transpose everything to
    the key of C.)"""
    
    print("Loading MIDI File:", file)

    midi = converter.parse(file)
    parts = instrument.partitionByInstrument(midi)  # will extract the parsed
                                                    # data separated into
                                                    # instructions for different
                                                    # instruments

    for part in parts:
        # There is one viable part per song, we'll return this one when found
        notes_to_parse = part.recurse()
        notes = []        # List to store tuples where the first element is
                          # either a '.'-separated string representing the notes
                          # being played at one time and the second element is
                          # a float representing the duration in seconds
                          
        keys = []         # A list will contain the musical keys in the song
        
        bpm = None        # Beats per minute, this changes a lot throughout the
                          # song. Used to scale durations.
                    
        offset = 0        # the offset is the number (float ) of beats (AKA
                          # quarter notes) into a song the notes we are
                          # currently reading in begin at
                          
        last_offset = 0   # keeps track of the offset out to which we've alread
                          # written data from
        
        all_notes_ato = []# All notes At This Offset (ATO). Tracks the notes
                          # that need to be added in between two offsets

        for element in notes_to_parse: 

            if (getattr(element, '__module__', None) == instrument.__name__):  
                continue       #Iignore the instrument type. Could enforce that
                               # it must be piano type but one of the files is
                               # misclassed.
            if (isinstance(element, note.Rest)):
                continue       # Ignore rests in the file, will infer them from
                               # from the offsets and durations
                    
            if (isinstance(element, key.Key)):                
                
                if (keys):     # This is a key change. Will later treat these as
                               # separators to split song into multiple songs
                               # each with one key).
                    notes.append('key change')
                    
                # Store the keys in order of appearance                       
                keys.append(str(element))
                
            elif (isinstance(element, tempo.MetronomeMark)):  # Update bpm 
                bpm = element.number  
                
            else: # Should be either a note or chord
                
                if (bpm is None):
                    print('bpm is None before first note or chord, skipping '\
                          + 'part')
                    notes.clear()
                    break          
                    
                if (element.offset == last_offset):
                # We're still at this offset, so keep adding to all_notes_ato
                    
                    if (isinstance(element, note.Note)):     
                        all_notes_ato.append((str(element.pitch), \
                                              element.duration.quarterLength))
                        
                    elif (isinstance(element, chord.Chord)):
                        all_notes_ato.append(('.'.join(str(n) for n in \
                            element.pitches), element.duration.quarterLength))                                   
                else:    # a new offset, we need to write all the different
                         # piano states that occurred in this offset interval,
                         # and add a rest if the offset interval is longer than
                         # the durations of any of the notes or chords
                    offset = element.offset    
                    cur_offset = last_offset        
                    
                    if (all_notes_ato):  # We have notes to write at this offset

                        # sort by duration
                        all_notes_ato.sort(key = lambda x: x[1])
                        while(cur_offset < offset):  
                            
                            shortest_duration = all_notes_ato[0][1]
                            if (shortest_duration < (offset - cur_offset)):
                                # write some intermediate lines, for those notes
                                # whose durations fall in this offset interval
                                
                                notes.append(('.'.join(n[0] for n in \
                                    all_notes_ato), shortest_duration * (60 /\
                                    bpm)))    # convert to seconds
                                
                                cur_offset += shortest_duration
                                
                                # Now get rid of those notes we just wrote
                                while(all_notes_ato and all_notes_ato[0][1] == \
                                      shortest_duration):
                                    all_notes_ato.pop(0)
                                    
                                if (not all_notes_ato):
                                    # If there are no notes to write and there
                                    # is still a gap between offset and
                                    # cur_offset, add a rest to complete the
                                    # interval
                                    notes.append(('rest', (offset - cur_offset) \
                                                  * (60 / bpm)))
                                    cur_offset = offset
                                    break
                                    
                            elif (shortest_duration > ((offset - cur_offset) + \
                                                       time_tol)):  
                                # All notes leftover should be transferred, but
                                # with their durations shortened.
                                # Added tolerance because of rounding errors.
                                corrected = []
                                for i in range(len(all_notes_ato)):
                                    corrected.append((all_notes_ato[i][0], \
                                           all_notes_ato[i][1] - (offset - \
                                                               cur_offset)))
                                all_notes_ato = corrected
                                cur_offset = offset
                                
                            else:  # they are equal (or close enough!)
                                cur_offset = offset
                                notes.append(('.'.join(n[0] for n in \
                                    all_notes_ato), all_notes_ato[0][1] * (60 /\
                                                bpm)))   # convert to seconds
                                # get ready for next offset interval
                                all_notes_ato.clear()
                               
                    # Read the note or chord
                    if (isinstance(element, note.Note)):
                        all_notes_ato.append((str(element.pitch), \
                                                element.duration.quarterLength))
                    elif (isinstance(element, chord.Chord)):
                        all_notes_ato.append(('.'.join(str(n) for n in \
                            element.pitches), element.duration.quarterLength))
                    last_offset = element.offset

        # In this dataset, there is just one viable part per song.
        # So once we get create a notes array from it.
        if (notes):
            return keys, notes
    
    return -1   # failure to return correct a viable song

def split_songs(keys, songs): 
    """Split songs with key changes into separate 'songs'. The significance 
     of a 'song' hereafter is that it will define a sequence from which training 
     sequences will be drawn using a window function. The window function will 
     not cross boundaries between songs so that each training sequence is
     entirely in one key (which will be transposed to C major)"""

    keys_by_song = []    # list of strings representing the key, to be returned
    notes_by_song = []   # list of list of tuples representing the note
                         # sequence, to be returned.
    for i in range(len(keys)):
        new_keys, notes = keys[i], songs[i]
        key_index = 0
        new_notes = []            # keeps track of the note arrays in between
                                  # each key change (and song end)
        for note in notes:
            
            cur_key = new_keys[key_index]
            if (note == 'key change'):
                if (new_keys[key_index + 1] == cur_key):  # key change to same 
                    key_index += 1                        # key, ignore.
                    continue
                if (new_notes):   # otherwise, save to *by_song arrays   
                    notes_by_song.append(new_notes.copy())
                    new_notes.clear()
                    keys_by_song.append(cur_key)
                key_index += 1                      
            else:
                new_notes.append(note)
        if (new_notes):     # Because the sequences don't end in key changes,
                            # we will likely have something leftover to save
            notes_by_song.append(new_notes.copy())
            keys_by_song.append(new_keys[key_index])
    return keys_by_song, notes_by_song

def note_to_piano_idx(a_note, n_keys_piano = 88):
    """Simply convert a note in the format {Letter}{Octave} to the 0-based index
    number of its corresponding position on the piano. Returns -1 for invalid
    input."""

    # relative offsets from C in the right-dir
    rel_offset = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}  
    piano_offset = 3  # C1 is the 4th key on the piano (index of 3)

    max_octave = 8    # Last key on the piano is C8
    notes_in_octave = 12

    try:
        a_note, octave = a_note[:-1], int(a_note[-1])
    except(ValueError):
        print('ERROR: invalid octave ', a_note[-1])
        return -1

    if (a_note[0] not in rel_offset):
        print('ERROR: invalid note letter ', a_note[0])
        return -1
    if (len(a_note) > 1):
        if (a_note[1] == '-'): # a flat!
            if (a_note[0] in ['C', 'F']):
                print('ERROR: no such note as ', a_note[:2])
                return -1
            idx = piano_offset + rel_offset[a_note[0]] + notes_in_octave * \
                   (octave - 1) - 1
        elif (a_note[1] == '#'): # a sharp!
            if (a_note[0] in ['B', 'E']):
                print('ERROR: no such note as ', a_note[:2])
                return -1                
            idx = piano_offset + rel_offset[a_note[0]] + notes_in_octave * \
                   (octave - 1) + 1
        else:
            print('ERROR: invalid addon for input ', a_note)
            return -1
    else:
        idx = piano_offset + rel_offset[a_note[0]] + notes_in_octave * (octave \
           - 1)
    if (0 <= idx <= n_keys_piano - 1):
        return idx
    else:
        print('ERROR: piano key index {} out of range'.format(idx))
        return -1


def songs_to_sequences(songs, time_tol = 1.e-3, n_keys_piano = 88):
    """Converts the sequences within songs to vector format (an n_keys_piano + 1th
    element NumPy Ndarray where the last element is the normalized duration (in
    quarter notes), and the rest of the elements are 1 for key on and 0 for key
    off.)"""
    sequences = []
    indices = None
    for song in songs:
        sequence = []
        durations = []
        for element in song:
            vector = np.zeros(n_keys_piano) # The current boolean array storing
                                            # which keys are being pressed. Will
                                            # add an additional n_keys_piano +
                                            # 1th element for the duration
                                            
            cur_note, duration = element    # separate note and duration (in
                                            # seconds)
            if (duration < time_tol):       # There are some rests with near
                continue                    # zero duration. Skip them.
            if ('.' in cur_note):           # chord
                notes = cur_note.split('.')
                for cur_note in notes:   
                    vector[note_to_piano_idx(cur_note)] = 1  
            elif (cur_note != 'rest'):      # a note
                vector[note_to_piano_idx(cur_note)] = 1
            sequence.append(vector)
            durations.append(float(duration))
        sequence = np.array(sequence)
        sequence = np.insert(sequence, len(sequence[0]), durations, axis = 1)
        sequences.append(sequence)        
                              
    return np.array(sequences)

def transpose_sequence(note_sequence, transposition):
    """Perform a right-shift on the keys part of the vectors. Effectively, this
    outputs a new sequence repesenting a song but transposed. The size of the
    shift is transposition."""
    
    if (transposition == 0):
        return note_sequence
    
    transposed_note_sequence = []
    for i in range(len(note_sequence)):
        transposed_note_sequence.append(np.concatenate((note_sequence[i]\
                [-transposition:], note_sequence[i][:-transposition])))
        
    return transposed_note_sequence

def transpose_sequences(sequences, keys_by_song):
    """Transpose (rightward) all sequences in sequences to the key of C major"""
    
    # relative offsets from C in the right-dir
    right_offset = {'C': 0, 'D': 10, 'E': 8, 'F': 7, 'G': 5, 'A': 3, 'B': 1}

    transposed_sequences = [] 
    for i in range(len(sequences)):
        notes, durations = sequences[i][:, :-1], sequences[i][:, -1]
        song_key = keys_by_song[i].split()[0]
        transposition = right_offset[song_key[0]]
        if (len(song_key) > 1):
            if (song_key[1] == '-'):
                transposition += 1
            elif (song_key[1] == '#'):
                transposition -= 1
            else:
                print('Problem with song_key: ', song_key)
                return
        transposed_sequence = transpose_sequence(notes, transposition)
        transposed_sequences.append(np.insert(transposed_sequence,\
                                    n_keys_piano, durations, axis = 1))   
    return transposed_sequences

def sequences_to_inputs(sequences, window_size = 16):
    """Apply a window function of size window_size across the dataset to create
    X. The next vector in the sequence is appended to y for each window. Returns
    X, y."""
    
    X = []
    y = []
    
    for i in range(len(sequences)):
        if (len(sequences[i]) < window_size + 1):
            print('Skipping index ', i, ' because the song is too short. Try' +\
                  'a shorter window_size to include it.')
            continue
        for j in range(len(sequences[i]) - window_size):
            X.append(sequences[i][j:j + window_size])
            y.append(sequences[i][j + window_size])

    return np.array(X), np.array(y)





