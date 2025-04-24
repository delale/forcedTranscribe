from typing_extensions import Union
import torch
import transformers
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


# Set device
def device_map_settings(user_opt_device: str = "auto"):
    if user_opt_device == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.mps.is_available():
            return "mps"
    else:
        return user_opt_device


# Create pipeline
def create_pipeline(
    model_id: str,
    device_map: Union[torch.device, str],
    torch_dtype: Union[torch.dtype, str] = "auto",
    **pipeline_kwargs,
) -> transformers.pipelines.automatic_speech_recognition.AutomaticSpeechRecognitionPipeline:
    # Init model
    model = AutoModelForSpeechSeq2Seq(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        device_map=device_map,
    )

    # Init processor
    processor = AutoProcessor.from_pretrained(model_id)

    # return pipeline
    return pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **pipeline_kwargs,
    )


# Directly from nyrahealth/CrisperWhisper
def adjust_pauses_for_hf_pipeline_output(pipeline_output, split_threshold=0.12):
    """
    Adjust pause timings by distributing pauses up to the threshold evenly between adjacent words.
    """

    adjusted_chunks = pipeline_output["chunks"].copy()

    for i in range(len(adjusted_chunks) - 1):
        current_chunk = adjusted_chunks[i]
        next_chunk = adjusted_chunks[i + 1]

        current_start, current_end = current_chunk["timestamp"]
        next_start, next_end = next_chunk["timestamp"]
        pause_duration = next_start - current_end

        if pause_duration > 0:
            if pause_duration > split_threshold:
                distribute = split_threshold / 2
            else:
                distribute = pause_duration / 2

            # Adjust current chunk end time
            adjusted_chunks[i]["timestamp"] = (current_start, current_end + distribute)

            # Adjust next chunk start time
            adjusted_chunks[i + 1]["timestamp"] = (next_start - distribute, next_end)
    pipeline_output["chunks"] = adjusted_chunks

    return pipeline_output
