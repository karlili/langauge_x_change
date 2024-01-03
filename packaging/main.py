from typing import Union

from fastapi import FastAPI



app = FastAPI()


@app.get("/")
def read_root():
    print("something changed 1")
    return {"Hello": "World"}

""" 
todo:
1. to accept file upload
2. to accept a default folder as the source of sound. - defined with docker volume

consider this resource
https://stackoverflow.com/questions/63048825/how-to-upload-file-using-fastapi

"""

from fastapi import File, UploadFile
from transcription.transcribe import transcribe

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open('/app/upload/'+file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()


    result = transcribe(
     './model/whisper-small-cantonese_23-12-2023-2157/checkpoint-400',
     './model/whisper-small-cantonese_23-12-2023-2157',
    # "openai/whisper-large-v3",
    # "openai/whisper-large-v3",
    '/app/upload/'+file.filename)

    
    #Remove the uploaded file afterwards
    

    return {"message": f"Successfully uploaded and processed {file.filename}",
    "result": f"{result}"}