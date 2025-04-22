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
            "segments"["words"]: (if `word_level`) word level transcriptions.
        Caveats:
            - "segments" and "words" must be dictionaries with also "start" and "end" keys.
            - "words" is a dictionary in the "segments" dictionary.
        This dictionary follows the output structure of OpeanAI-Whisper.
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
        if "words" not in transcription["segments"][0]:
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
    if segment_level or word_level:
        i, j = 0, 0
        same_boundary_seg = 0
        same_boundary_word = 0

        # Loop through segments
        for ii, segment in enumerate(transcription["segments"]):
            # Segment level
            if segment_level:
                interval_num = i + 1
                t0 = segment["start"]
                t1 = segment["end"]

                # Add segment if not same boundary
                if not t1 >= duration and not t0 == t1:
                    same_boundary_seg = 0
                    praat.call(tg, "Insert boundary", 2, t1)
                    text = segment["text"]
                elif t0 == t1 and ii > 0:
                    # Adding to "prev" boundary
                    interval_num -= 1
                    same_boundary_seg += 1
                    text = " ".join(
                        [
                            transcription["segments"][x]["text"]
                            for x in range(ii - same_boundary_seg, ii)
                        ]
                    )
                    +segment["text"]
                elif t0 == t1 and ii == 0:
                    praat.call(
                        tg,
                        "Insert boundary",
                        2,
                        t1 + 0.001,
                    )
                    text = segment["text"]

                # Add text
                praat.call(
                    tg,
                    "Set interval text",
                    2,
                    interval_num,
                    text,
                )
                i += 1

            # Word level
            if word_level:
                for jj, word in enumerate(segment["words"]):
                    interval_num = j + 1
                    t0 = word["start"]
                    t1 = word["end"]

                    # Add word if not same boundary
                    if not t1 >= duration and not t0 == t1:
                        same_boundary_word = 0
                        praat.call(tg, "Insert boundary", word_ntier, t1)
                        text = word["word"]
                    elif t0 == t1 and jj > 0:
                        # Adding to "prev" boundary
                        interval_num -= 1
                        same_boundary_word += 1
                        text = " ".join(
                            [
                                segment["words"][x]["word"]
                                for x in range(ii - same_boundary_word, jj)
                            ]
                        )
                        +segment["text"]
                    elif t0 == t1 and jj == 0:
                        praat.call(
                            tg,
                            "Insert boundary",
                            word_ntier,
                            t1 + 0.001,
                        )
                        text = segment["word"]

                    # Add text
                    praat.call(
                        tg,
                        "Set interval text",
                        word_ntier,
                        interval_num,
                        text,
                    )
                    j += 1

    return tg
