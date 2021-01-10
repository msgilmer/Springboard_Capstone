# Springboard_Capstone
Music Generation with LSTM network trained the music of Chopin ([Kaggle dataset](https://www.kaggle.com/soumikrakshit/classical-music-midi))

The model architecture and encoding largely follow the process described by this [paper](https://www.tandfonline.com/doi/full/10.1080/25765299.2019.1649972) in which a similar network is trained on the music of Bach. However, there are two other important differences:
  1. Instead of the data augmentation method used by the paper (randomized transpositions of songs to different keys), I transpose all songs to the key of C Major)
      - One advantage is shorter training while still containing the same amount of musical information
      - Another is a simplification of the problem the neural network needs to solve
      - One can still transpose the generated music to any key following inference
  2. Added custom loss function which penalizes for the error in a duration prediction (in addition to error in the prediction of notes and chords)
      - I haved named this function the <b>Maestro Loss</b> which can be tuned with a paramter (<b>harshness</b>). A higher harshness gives more weight to the duration prediction
      - More details in ./notebooks/model_training.ipynb

    .
    ├── chopin                  # Contains MIDI files downloaded from Kaggle (link above)
    │   ├── *.mid
    ├── docs                    
    │   ├── Capstone_Proposal.pdf
    │   ├── deployment_strategy
    ├── midi_output             # Contains MIDI files generated from validation data (which is used as music generation input) or music generated from a model on such an input
    │   ├── *.mid
    ├── model_data              # Contains .csv files with training performance data (loss and metrics vs. Epochs). Saved in ./notebooks/model_training.ipynb and read-in by visualize_performance.ipynb
    │   ├── *.mid
    ├── models                  # Contains .h5 files with trained models. Saved in ./notebooks/model_training.ipynb and best model read-in by ./web_app/ChopinBotApp.py
    │   ├── *.h5
    ├── notebooks               # Contains Jupyter notebooks. They are meant to be run/understood sequentially in the order they appear below (also alphabetical)
    │   ├── data_read_and_process.ipynb
    │   ├── model_training.ipynb
    │   └── visualize_performance.ipynb
    ├── train_and_val           # Contains .pkl files of the training and validation partitions of the processed dataset (./notebooks/data_read_and_process.ipynb)
    │   ├── X_train.pkl
    │   ├── y_train.pkl 
    │   ├── X_val.pkl 
    │   └── y_val.pkl 
    ├── webapp                  # Contains web application code using [Streamlit](https://www.streamlit.io/). Run as 'streamlit run ChopinBotApp.py'
    │   ├── ChopinBotApp.py 
    │   └── custom_funcs.py     # Custom functions written in keras backend language. Necessary to load a trained model file (.h5) from ../models
    └── ...
    


