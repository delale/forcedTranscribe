import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

torch_dtye = torch.float16 if torch.cuda.is_available() else torch.float32
