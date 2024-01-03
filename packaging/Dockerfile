FROM python:3.9.18-bullseye

RUN apt update && apt upgrade -y
RUN apt install ffmpeg -y

WORKDIR /app

# RUN python -m venv /app/language_x_change/venv
# RUN source language_x_change/venv/bin/activate

# COPY model ./model
# COPY requirements.txt .
COPY . .
RUN mkdir /app/upload


#Creating a virtual environment and installing dependencies
RUN python -m venv ./venv
RUN . venv/bin/activate


#Install the python dependencies
# RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000