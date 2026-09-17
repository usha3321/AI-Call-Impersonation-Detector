import librosa

from src.feature_extraction import extract_mfcc
from src.similarity import calculate_similarity


def load_audio(file_path):
    audio, sample_rate = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    return audio, sample_rate


if __name__ == "__main__":
    audio, sample_rate = load_audio(
        "data/samples/reference.wav"
    )

    print("Audio loaded successfully!")
    print("Sample rate:", sample_rate)
    print("Number of samples:", len(audio))
    print("Duration in seconds:", len(audio) / sample_rate)

    mfcc_features = extract_mfcc(
        audio,
        sample_rate
    )

    print("MFCC features extracted successfully!")
    print("MFCC feature shape:", mfcc_features.shape)
    print("MFCC values:", mfcc_features)
    test_audio, test_sample_rate = load_audio(
        "data/samples/test_voice.wav"
    )

    test_mfcc = extract_mfcc(
        test_audio,
        test_sample_rate
    )

    similarity_score = calculate_similarity(
        mfcc_features,
        test_mfcc
    )

    print("Voice similarity score:", similarity_score)    