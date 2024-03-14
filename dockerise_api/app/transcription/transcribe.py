from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import datetime
import json

def transcribe( model_path, processor_path, audio_file_path):
  model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_path, 
    # local_files_only=True,
     low_cpu_mem_usage=True,
     use_safetensors=True,
    #  cache_dir="/Volumes/BACKUP/Coding/HUGGING_FACE/models"
  )

  processor = AutoProcessor.from_pretrained(processor_path)

  transcriber = pipeline("automatic-speech-recognition", 
      model=model,  
      tokenizer=processor.tokenizer,
      feature_extractor=processor.feature_extractor,
      chunk_length_s=20,
      max_new_tokens=256,
      # batch_size=16,
      return_timestamps=True,
      generate_kwargs={"task": "transcribe", "language": "cantonese"},
    )
  # transcriber.tokenizer.get_decoder_prompt_ids(language='cantonese', task="transcribe")

  result = transcriber(audio_file_path)
  return result
  