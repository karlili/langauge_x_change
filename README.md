# Prerequisite 

Steps to setup

- Make sure you have python install in your system
    - Homebrew (Package installer for mac ) https://brew.sh/
    - Run brew install ffmpeg (this package helps to read the audio mp3 files)

Create a virtual environment
 - Navigate to the project directory `cd <PATH>/language_x_change`
 - Run `python3 -m venv <name_of_virtualenv>` to create a place to hold all the required libraries needed to run the project
 - Use the environemnt we just created as a workspace by running `source <name_of_virtualenv>/bin/activate`
 - Run `pip install -r requirements.txt` 
