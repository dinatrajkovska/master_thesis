### IMPORTANT: this code might contain bugs! Go through each line of code and verify if it does the right thing!

### To be able to import the necessary packages, you can do one of two things:
### 1. Change your .bashrc file so you can import globally installed packages (such as TensorFlow)
### 2. Install the packages locally
### How to do this is described in the speech wiki. If you have questions about this, just ask me.
from __future__ import division
import os
import soundfile
import numpy as np
import librosa
import tensorflow as tf

### Firstly, you need to load your data. In this example, we load the ESC-10 data set.
### The name of each file in the directory for this data set follows a pattern:
### {fold number}-{file id}-{take number}-{target class number}
### You can ignore take number and file id for now.
### The fold number indicates which test fold the file belongs to, in order to be able to perform cross-validation.
### You can look up information about cross-validation online.
### Now, for simplicity, we take folds 1-3 as train folds, 4 as validation fold and 5 as test fold.
### The target class number indicates which sound is present in the file.
directory_path = "/users/spraak/wboes/spchtemp/data/esc10"
filenames = os.listdir(directory_path)
train_log_mel_spectrograms = []
train_delta_log_mel_spectrograms = []
train_target_numbers = []
val_log_mel_spectrograms = []
val_delta_log_mel_spectrograms = []
val_target_numbers = []
test_log_mel_spectrograms = []
test_delta_log_mel_spectrograms = []
test_target_numbers = []

### Loop over all files
for filename in filenames:
    ## Find fold number
    fold_number = int(filename[0])
    ## Extract mel spectrogram
    # Read file
    (audio, sr) = soundfile.read(directory_path + "/" + filename)
    # Convert stereo to mono if necessary
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    # Extract stft
    spectrogram = librosa.core.stft(audio, n_fft=1024, hop_length=512, center=False)
    # Convert to mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        S=np.abs(spectrogram) ** 2, n_fft=1024, hop_length=512, n_mels=128
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    log_mel_spectrogram = log_mel_spectrogram.T
    log_mel_spectrogram = np.array(log_mel_spectrogram, dtype=np.float32)
    # Normalize
    log_mel_spectrogram = (log_mel_spectrogram - np.min(log_mel_spectrogram)) / (
        np.max(log_mel_spectrogram) - np.min(log_mel_spectrogram)
    )
    # Compute delta log mel spectrogram
    delta_log_mel_spectrogram = librosa.feature.delta(log_mel_spectrogram, axis=0)
    # Find target class number
    target_number = int(filename[-5])
    # Depending on the fold number, use file to train, validate or test model
    if fold_number == 4:
        val_log_mel_spectrograms.append(log_mel_spectrogram)
        val_delta_log_mel_spectrograms.append(delta_log_mel_spectrogram)
        val_target_numbers.append(target_number)
    elif fold_number == 5:
        test_log_mel_spectrograms.append(log_mel_spectrogram)
        test_delta_log_mel_spectrograms.append(delta_log_mel_spectrogram)
        test_target_numbers.append(target_number)
    else:
        train_log_mel_spectrograms.append(log_mel_spectrogram)
        train_delta_log_mel_spectrograms.append(delta_log_mel_spectrogram)
        train_target_numbers.append(target_number)

### Convert data to numpy arrays and ensure data has correct format
train_log_mel_spectrograms = np.array(train_log_mel_spectrograms)
train_log_mel_spectrograms = np.expand_dims(train_log_mel_spectrograms, axis=-1)
train_delta_log_mel_spectrograms = np.array(train_delta_log_mel_spectrograms)
train_delta_log_mel_spectrograms = np.expand_dims(
    train_delta_log_mel_spectrograms, axis=-1
)
train_target_numbers = np.array(train_target_numbers)
val_log_mel_spectrograms = np.array(val_log_mel_spectrograms)
val_log_mel_spectrograms = np.expand_dims(val_log_mel_spectrograms, axis=-1)
val_delta_log_mel_spectrograms = np.array(val_delta_log_mel_spectrograms)
val_delta_log_mel_spectrograms = np.expand_dims(val_delta_log_mel_spectrograms, axis=-1)
val_target_numbers = np.array(val_target_numbers)
test_log_mel_spectrograms = np.array(test_log_mel_spectrograms)
test_log_mel_spectrograms = np.expand_dims(test_log_mel_spectrograms, axis=-1)
test_delta_log_mel_spectrograms = np.array(test_delta_log_mel_spectrograms)
test_delta_log_mel_spectrograms = np.expand_dims(
    test_delta_log_mel_spectrograms, axis=-1
)
test_target_numbers = np.array(test_target_numbers)

### Keras provides multiple ways of defining a model.
### One option is to create a Sequential model.
model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=(1, 3), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=(1, 3), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same"),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=(5, 1), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=(5, 1), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=(1, 1), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.MaxPool2D(pool_size=(4, 1), padding="same"),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(1, 3), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(1, 3), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.MaxPool2D(pool_size=(1, 2), padding="same"),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(5, 1), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(5, 1), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=(1, 1), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.MaxPool2D(pool_size=(4, 1), padding="same"),
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(5, 3), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(5, 3), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=128, kernel_size=(1, 1), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.MaxPool2D(pool_size=(4, 2), padding="same"),
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=(5, 3), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=(5, 3), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=(1, 1), strides=(1, 1), padding="same"
        ),
        tf.keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.MaxPool2D(pool_size=(4, 2), padding="same"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

### Compile the model, i.e., choose an optimizer, loss function and metric(s)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

### Fit the training data for a number of epochs
model.fit(
    np.concatenate(
        (train_log_mel_spectrograms, train_delta_log_mel_spectrograms), axis=-1
    ),
    train_target_numbers,
    epochs=125,
    verbose=1,
)

### Find optimal prediction threshold based on validation data
optimal_threshold = None
optimal_accuracy = 0
for threshold in np.arange(0.1, 0.99, 0.05):
    _, val_acc = model.evaluate(
        np.concatenate(
            (val_log_mel_spectrograms, val_delta_log_mel_spectrograms), axis=-1
        ),
        val_target_numbers,
        verbose=0,
    )
    if val_acc > optimal_accuracy:
        optimal_threshold = threshold

### Calculate test accuracy
_, test_acc = model.evaluate(
    np.concatenate(
        (test_log_mel_spectrograms, test_delta_log_mel_spectrograms), axis=-1
    ),
    test_target_numbers,
    verbose=0,
)
print("\nTest accuracy: {0}%".format(test_acc * 100))

### Other things to do: data augmentation, apply early stopping, saving/loading data and models, ensembling etc.
