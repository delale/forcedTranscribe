import logging
import parselmouth
from parselmouth import praat


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
