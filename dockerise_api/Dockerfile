FROM python:3.9.18-bullseye

RUN apt update && apt upgrade -y
RUN apt install tmux vim htop ffmpeg -y

WORKDIR /app
ENV HUGGING_FACE_HUB_TOKEN=''

# RUN python -m venv /app/language_x_change/venv
# RUN source language_x_change/venv/bin/activate

# COPY model ./model
# COPY requirements.txt .
COPY . .
RUN mkdir /app/upload


#Creating a virtual environment and installing dependencies
# RUN python -m venv ./venv
# RUN . venv/bin/activate


#Install the python dependencies
# RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn main:app --reload --host 0.0.0.0
EXPOSE 8000
