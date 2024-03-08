from typing import Union

from fastapi import FastAPI
from fastapi.responses import StreamingResponse


app = FastAPI()


@app.get("/")
def read_root():
    # print("something changed 1")
    return {"status": "Running.... Welcome language-x-change"}

""" 
todo:
1. to accept file upload
2. to accept a default folder as the source of sound. - defined with docker volume

consider this resource
https://stackoverflow.com/questions/63048825/how-to-upload-file-using-fastapi

"""
import os
from fastapi import File, UploadFile
from transcription.transcribe import transcribe
from difflib import SequenceMatcher, Differ

@app.post("/transcribe")
def upload_audio(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open('/app/upload/'+file.filename, 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()


    # just keep it simple and use whisper model for now
    result = transcribe(
        "openai/whisper-large-v3",
        "openai/whisper-large-v3",
        '/app/upload/'+file.filename)

    # remove the file when processing is done
    try:
        os.remove('/app/upload/'+file.filename)
        print(f"File /app/upload/'{file.filename}' has been deleted.")
    except FileNotFoundError:
        print(f"Error: File /app/upload/'{file.filename}' not found.")

    return {
        # "message": f"Successfully processed {file.filename}",
        "fileName": file.filename,
        "transcription": f"{result['text']}"
        }


@app.post('/compare')
def diff(input, expected):
    s = SequenceMatcher(None, input, expected)
    d = Differ()
    match_ratio = s.ratio()
    difference = d.compare(input, expected)

    # print( f"{difference}" )
    character_by_character_diff = []

    for index, char in enumerate(difference):
        
        if "-" in char:
            # print(f"Item '{char}' contains a hyphen.")

            key = char.replace("-", "").strip()
            if key in expected:
                character_by_character_diff.append( {  "expected_char": key, "contains": False})

        elif "+" in char:
            key = char.replace("+", "").strip()
            if key in expected:
                character_by_character_diff.append( { "expected_char": key, "contains": False})
        
        else:
            # print(f"Item '{char}' does not contain a hyphen.")
            key = char.strip()
            character_by_character_diff.append( {  "expected_char": key, "contains":  True })


    return {
        "input": f"{input}",
        "expected": f"{expected}",
        "match_ratio": f"{match_ratio}",
        "char_by_char_diff": character_by_character_diff,
        # "difference": difference,
        
        }
