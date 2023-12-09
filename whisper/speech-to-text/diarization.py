'''
Install the following dependencies before running the script

- pip install pyannote.audio


'''


from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="<HUGGING_FACE_USER_TOKEN>")

diarization = pipeline("sample.mp3")

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)
