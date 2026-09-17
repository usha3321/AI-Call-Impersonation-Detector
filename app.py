import streamlit as st
import tempfile
import os

from src.audio_processing import load_audio
from src.feature_extraction import extract_mfcc
from src.similarity import calculate_similarity
"""What this code does

librosa loads the audio file.

sr=16000 converts the audio to a standard sample rate.

mono=True converts the audio into one audio channel.

audio contains the sound data.

sample_rate tells us how many audio samples are processed per second."""


st.set_page_config(
    page_title="SecureCall AI",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ SecureCall AI")

st.write(
    "Voice Similarity and Verification Prototype"
)

st.info(
    "This prototype compares two voice recordings "
    "using MFCC features and cosine similarity."
)


reference_file = st.file_uploader(
    "Upload Reference Voice",
    type=["wav"],
    key="reference_voice"
)

test_file = st.file_uploader(
    "Upload Test Voice",
    type=["wav"],
    key="test_voice"
)


if st.button("Compare Voices"):

    if reference_file is None or test_file is None:

        st.warning(
            "Please upload both WAV voice recordings."
        )

    else:

        reference_path = None
        test_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as reference_temp:

                reference_temp.write(
                    reference_file.getvalue()
                )

                reference_path = reference_temp.name


            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as test_temp:

                test_temp.write(
                    test_file.getvalue()
                )

                test_path = test_temp.name


            reference_audio, reference_sr = load_audio(
                reference_path
            )

            test_audio, test_sr = load_audio(
                test_path
            )


            reference_mfcc = extract_mfcc(
                reference_audio,
                reference_sr
            )

            test_mfcc = extract_mfcc(
                test_audio,
                test_sr
            )


            similarity_score = calculate_similarity(
                reference_mfcc,
                test_mfcc
            )

            similarity_percentage = (
                similarity_score * 100
            )


            st.success(
                "Voice comparison completed!"
            )

            st.metric(
                "Voice Similarity Score",
                f"{similarity_percentage:.2f}%"
            )


            if similarity_score >= 0.90:

                st.success(
                    "The recordings have high similarity."
                )

            elif similarity_score >= 0.70:

                st.warning(
                    "The recordings have moderate similarity."
                )

            else:

                st.error(
                    "The recordings have low similarity."
                )


            st.caption(
                "This experimental score does not prove "
                "that a voice is genuine or fraudulent."
            )


        except Exception as error:

            st.error(
                f"An error occurred: {error}"
            )


        finally:

            if reference_path and os.path.exists(
                reference_path
            ):

                os.remove(reference_path)


            if test_path and os.path.exists(
                test_path
            ):

                os.remove(test_path)