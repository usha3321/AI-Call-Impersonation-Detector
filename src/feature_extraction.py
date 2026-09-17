import librosa
import numpy as np


def extract_mfcc(audio, sample_rate):
    mfcc_features = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    mfcc_mean = np.mean(
        mfcc_features,
        axis=1
    )

    return mfcc_mean