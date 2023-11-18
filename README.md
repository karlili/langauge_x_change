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

To run the notebook environment, run this `jupyter notebook` in the same terminal like this. In the end, it will show you the link to open up the notebook

```
(<name_of_virtualenv>) kenny@macbookpro langauge_x_change % > jupyter notebook
...
...


[I 2023-11-18 11:42:05.838 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2023-11-18 11:42:05.840 ServerApp] 
    
    To access the server, open this file in a browser:
        file:///Users/kenny/Library/Jupyter/runtime/jpserver-25640-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/tree?token=645564ca969d88ce57437d3a5523f080c282d93f95084375
        http://127.0.0.1:8888/tree?token=645564ca969d88ce57437d3a5523f080c282d93f95084375
[I 2023-11-18 11:42:05.851 ServerApp] Skipped non-installed server(s): bash-language-server, dockerfile-language-server-nodejs, javascript-typescript-langserver, jedi-language-server, julia-language-server, pyright, python-language-server, python-lsp-server, r-languageserver, sql-language-server, texlab, typescript-language-server, unified-language-server, vscode-css-languageserver-bin, vscode-html-languageserver-bin, vscode-json-languageserver-bin, yaml-language-server

``` 