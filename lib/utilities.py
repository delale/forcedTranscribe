"""General utilities."""

import logging
from typing_extensions import Tuple
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import parselmouth
from parselmouth import praat


def load_audio(audio_path: str) -> Tuple[np.ndarray, float]:
    """Loads and preprocess audio file.

    Parameters
    ----------
    audio_path : str
        Path to audio file.

    Returns
    -------
    tuple
        np.ndarray:
            Audio waveform.
        float
            Audio duration in seconds.
    """
    y, sr = torchaudio.load(uri=audio_path, normalize=True).numpy()

    # Check type
    if not y.dtype == np.dtype("float32"):
        y = y.astype(np.float32)

    # Normalise
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = (y - y_mean) / y_std

    # Resample to 16000 Hz
    resampler = T.Resample(sr, 16000)
    waveform = resampler(
        torch.unsqueeze(
            torch.tensor(y_norm / 8),
            0,
        )
    )

    return waveform, len(y) / sr


def textgrid_from_transcription(
    transcript: dict,
    duration: float,
) -> parselmouth.TextGrid:
    """Creates Praat TextGrid from transcription.

    Parameters
    ----------
    transcript : dict
        Transcription dictionary from transformers.pipelines.automatic_speech_recognition.AutomaticSpeechRecognitionPipeline:
            "text": full transcription.
            "chunks": list[dict]
                List of dictionaries containing "text" for each chunk (optionally also "timestamp").
    duration : float
        Audio file duration.

    Returns
    -------
    parselmouth.TextGrid
        TextGrid containing transcription (also segment level and word level if specified).
    """
    # Checks
    if not "chunks" in transcript:
        if "text" in transcript:
            logging.warning("Only 'text' available -> no boundaries.")
        else:
            raise ValueError(
                "`transcript` must be of type dict with keys 'text' and/or 'chunks'"
            )

    # Init TextGrid
    tg = parselmouth.TextGrid(
        start_time=0.0,
        end_time=duration,
        tier_names=["transcription"],
    )

    # If only "text" no boundaries
    if not "chunks" in transcript and "text" in transcript:
        praat.call(
            tg,
            "Set interval text",
            1,
            1,
            transcript["text"],
        )
        return tg

    # Else: add boundaries and labels from transcript
    prev_chunk_end = 0.0
    for chunk in transcript["chunks"]:
        t0, t1 = chunk["timestamp"]
        text = chunk["text"]

        if t0 == t1:
            # Add small gap
            t1 += 0.001

        if t0 != prev_chunk_end:
            # Add start boundary
            praat.call(
                tg,
                "Insert boundary",
                1,
                t0,
            )

        if t1 < duration:
            # Add end boundary
            t1 = duration
            praat.call(
                tg,
                "Insert boundary",
                1,
                t1,
            )

        # Get interval num
        i_interval = praat.call(
            tg,
            "Get interval at time",
            1,
            (t0 + t1) / 2,
        )

        # Set label
        praat.call(
            tg,
            "Set interval text",
            1,
            i_interval,
            text,
        )

        # Update end boundary
        prev_chunk_end = t1

    return tg
