# Springboard_Capstone
Music Generation with LSTM network trained the music of Chopin ([Kaggle dataset](https://www.kaggle.com/soumikrakshit/classical-music-midi))

The model architecture and encoding largely follow the process described by this [paper](https://www.tandfonline.com/doi/full/10.1080/25765299.2019.1649972) in which a similar network is trained on the music of Bach. However, there are two other important differences:
  1. Instead of the data augmentation method used by the paper (randomized transpositions of songs to different keys), I transpose all songs to the key of C Major)
      - One advantage is shorter training while still containing the same amount of musical information
      - Another is a simplification of the problem the neural network needs to solve
      - One can still transpose the generated music to any key following inference
      - The effect this has on the data is most easily understood from the following plot where the x-axis is the index of (0-87) of the piano key being played and the y-axis is the total # of times that key was played in the dataset (see ./notebooks/data_read_and_process.ipynb for the code and more details)
![](./images/original_vs_transposed.png)
  2. Added custom loss function which penalizes for the error in a duration prediction (in addition to error in the prediction of notes and chords via standard Binary Cross-entropy)
      - I haved named this function the <b>Maestro Loss</b> which can be tuned with a parameter (<b>harshness</b>). A higher harshness gives more weight to the duration prediction (see ./noteooks/model_training.ipynb for the code and more details):

![](./images/maestro_equation.jpg)
</br></br>
<b>Directory Structure</b>

    .
    ├── chopin                  # Contains MIDI files downloaded from Kaggle (link above)
    │   └── *.mid
    ├── docs                    
    │   ├── Capstone_Proposal.pdf
    │   └── deployment_strategy
    ├── images                  # Contains the images embedded above or in the web application
    │   ├── ChopinBotApp_flowchart.jpg
    │   ├── chopin.jpg    
    │   ├── maestro_equation.jpg
    │   ├── original_vs_transposed.png
    │   └── precision_and_recall.jpg
    ├── midi_output             # Contains MIDI files generated from validation data (which is used as music generation input) or music generated from a model on such an input
    │   └──  *.mid
    ├── model_data              # Contains .csv saved in ./notebooks/model_training.ipynb and ./notebooks/model_training_v2.ipynb (v2)
    │   ├── gradient_data       # Statistics on the gradients of the network over epochs
    │   │   └── gradient*.csv
    │   ├── performance_data    # Statistics on the gradients of the network over epochs   
    │   │   └── best_maestro_model*.csv
    │   ├── weight_data         # Statistics on the weights of the network over epochs
    │   │   └── weight*.csv     
    ├── models                  # Contains .h5 files with trained models. Saved in ./notebooks/model_training.ipynb (and v2) and best model read-in by ./web_app/ChopinBotApp.py
    │   └──  *.h5
    ├── notebooks               # Contains Jupyter notebooks. The commented numbers below indicate the order to read/run the notebooks for the prototype
    │   ├── data_read_and_process.ipynb     # Parse Midi files and prepare data for network input
    │   ├── data_read_and_process_v2.ipynb  # Same as above but with extended vectors with more nodes responsible for the duration prediction.
    │   ├── model_training.ipynb            # Model building, training, and exploration of gradients
    │   ├── model_training_v2.ipynb         # Same as above but with the data from v2 and an exploration of gradients and weights
    │   ├── prune_model.ipynb               # Prunes a pre-trained model
    │   ├── scaled_prototype.ipynb          # Shows that training works for a scaled dataset (via increasing window_size) (just over 1 GB)
    │   └── visualize_performance.ipynb     # Plots showing loss and metrics of models from ./model_training.ipynb over epochs
    ├── train_and_val           # Contains .npy files, but currently ignored with .gitignore (unless otherwise stated below) because of their large file size, of the training and validation partitions of the processed dataset (./notebooks/data_read_and_process.ipynb)
    │   ├── transposed_chopin_seqeunces.npy # Saved by ../notebooks/model_training.ipynb and loaded by ../notebooks/scaled_prototype.ipynb
    │   ├── X_train.npy            # The first four saved in ../notebooks/model_training.ipynb
    │   ├── y_train.npy 
    │   ├── X_val.npy 
    │   ├── y_val.npy 
    │   ├── X_train_ext.npy        # Next four saved in ../notebooks/model_training_v2.ipynb ('ext' is for extended)
    │   ├── y_train_ext.npy 
    │   ├── X_val_ext.npy
    │   ├── y_val_ext.npy
    │   ├── X_val_ext_1.npy        # Not in .gitignore. We use X_val sequences as input into the music generation in ../web_app. This set is broken into two to avoid Git LFS.
    │   └── X_val_ext_2.npy        # Not in .gitignore
    ├── web_app                 # Contains web application code using [Streamlit](https://www.streamlit.io/). Run as 'streamlit run ChopinBotApp.py' (Work in progress)
    │   ├── ChopinBotApp.py
    │   ├── logfiles                  # logfiles for each unique session. Used for monitoring errors
    │   │   └── logzero*log
    │   ├── rndm_seed_index_files     # txt files that uniquely store the last randomly-generated seed index (for the X_val set) for each unique session.
    │   │   └── rndm*txt
    │   ├── precision_and_recall.py   # script to generate a plot 9../images/precision_and_recall.jpg) for embedding into the application
    │   └── requirements.txt          # generated with pipreqs
    └── ...
    
