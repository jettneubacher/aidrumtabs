# Preprocess a .wav file and store feature info in 0.116ms long timeframes
# Feature info is stored in a numpy array with shape (10, 1039)
# 10 Timesteps per timeframe, 1039 total features

import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

def standardize(X):
    scaler = StandardScaler()

    # Flatten the data to apply scaling across all features
    X_reshaped = X.reshape(-1, X.shape[-1])

    # Fit the scaler on the training data and transform both train and test
    X_standardized = scaler.fit_transform(X_reshaped)

    # Reshape the standardized data back to the original shape (samples, time_steps, features)
    X_standardized = X_standardized.reshape(X.shape)

    return X_standardized


def preprocess_audio_file(file_path, time_steps=10, timeframe_length=0.116):
    #22050 sample rate for speed. Model was trained on this sample rate
    y, sr = librosa.load(file_path, sr=22050, mono=True, dtype=np.float32)


    hop_length = 256

    #Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    spec = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    #Determine length of song in respect to timeframes
    total_feature_frames = mfccs.shape[1]
    total_timeframes = total_feature_frames // time_steps

    X = []

    #Split features into timeframes that are 10 time steps long
    #10 time steps is equivalent to 0.116 milliseconds given a sample rate of 22050 and default hop length of 512
    for timeframe_index in range(total_timeframes):
        start_idx = timeframe_index * time_steps
        end_idx = start_idx + time_steps

        mfcc_frame = mfccs[:, start_idx:end_idx]
        spectrogram_frame = spec[:, start_idx:end_idx]
        onset_env_frame = onset_env[start_idx:end_idx].reshape(1, -1)
        
        feature_vector = np.concatenate([mfcc_frame.T, spectrogram_frame.T, onset_env_frame.T], axis=-1)  # Shape: (10, features)
        #print(f"Feature vector shape: {feature_vector.shape}")  # Should be (10, 1039)
        X.append(feature_vector)

    X = np.array(X, dtype=np.float32)

    X = standardize(X)

    X = X.astype(np.float16)

    return X

"""
Returns:
    X: np.ndarray of shape (num_timeframes, 10, 1039)
"""