FROM python:3.9.18-bullseye

RUN apt update && apt upgrade -y
RUN apt install tmux vim htop ffmpeg -y

WORKDIR /app


COPY . .

#Creating a virtual environment and installing dependencies

#Install the python dependencies
# RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT ["/bin/bash"]