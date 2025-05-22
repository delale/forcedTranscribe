"""Run MFA using subprocess."""


def force_align(
    path_to_corpus: str,
    output_path: str = None,
    speaker_characters: int = None,
    dictionary: str = "english_mfa",
    acoustic_model: str = "english_mfa",
    clean: bool = False,
    beam: int = 100,
    retry_beam: int = 400,
    include_original_text: bool = True,
    textgrid_cleanup: bool = True,
    num_jobs: int = 4,
):
    pass
