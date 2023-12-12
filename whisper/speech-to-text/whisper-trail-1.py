'''
Install the following dependencies before running the script

- pip install transformers accelerate
- pip install json datetime
- pip install torch torchvision torchaudio

'''

import torch
import json
import datetime

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device ="cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
torch_dtype=torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, 
    torch_dtype=torch_dtype, 
    low_cpu_mem_usage=True,
     use_safetensors=True,
    # cache_dir="/Volumes/BACKUP/Coding/HUGGING_FACE/models"
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=1000,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
    generate_kwargs={"language": "cantonese"}
)


# this is the place you modify your input - the name of the mp3 file you want to run
result = pipe("source/Audio1.mp3")

# then it will write the response in a json file named as the current date time
now = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
json_object = json.dumps(result, indent=4)
with open('output/'+now+".json", "w") as f:
    f.write(json_object)

# also it will print out the result in the following output block
print(result)