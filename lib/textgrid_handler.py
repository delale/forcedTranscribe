import logging
import parselmouth
from parselmouth import praat


def textgrid_from_transcription(
    transcription: dict,
    duration: float,
    segment_level: bool = False,
    word_level: bool = False,
) -> parselmouth.TextGrid:
    """Creates Praat TextGrid from transcription.

    Parameters
    ----------
    transcription : dict
        Transcription dictionary including the following keys:
            "text": full transcription.
            "segments": (if `segment_level`) segment level transcriptions (e.g. sentences).
            "words": (if `word_level`) word level transcriptions.
        Caveat: "segments" and "words" must be dictionaries with also "start" and "end" keys.
    duration : float
        Audio file duration.
    segment_level : bool
        Include segment level transcriptions. Default is False.
    word_level : bool
        Include word level transcriptions. Default is False.

    Returns
    -------
    parselmouth.TextGrid
        TextGrid containing transcription (also segment level and word level if specified).
    """
    # Init tier names
    if not isinstance(transcription, dict):
        raise ValueError("`transcription` must be of type dict.")
    if "text" not in transcription:
        raise ValueError("'text' not found in `transcription`.")
    tier_names: list = ["text"]
    if segment_level:
        if "segments" not in transcription:
            logging.warning(
                "No segment level transcription found but segment_level is True: unable to add 'segments' tier to TextGrid."
            )
            segment_level = False
        else:
            tier_names.append("segments")
    if word_level:
        if "words" not in transcription:
            logging.warning(
                "No word level transcription found but word_level is True: unable to add 'words' tier to TextGrid."
            )
            word_level = False
        else:
            tier_names.append("words")

    # Init TextGrid
    tg = parselmouth.TextGrid(
        start_time=0.0,
        end_time=duration,
        tier_names=tier_names,
    )

    # Add `text`
    praat.call(
        tg,
        "Set interval text",
        1,
        1,
        transcription["text"],
    )

    # Set appropriate tier numbers
    word_ntier = 3 if segment_level else 2

    # Add segment level
    if segment_level:
        pass
        # TODO: finish this; decide how to handle if word_level but not segment_level
