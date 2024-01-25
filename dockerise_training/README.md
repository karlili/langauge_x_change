language-x-change Training

this docker image helps to encapsulate the complexity in the training process, by using weights & biases (W&B) sweeps

# Prerequisite 

- You will need to have Docker running on your machine. It is available for MacOS and Window machine. You can download a copy from here (https://www.docker.com/products/docker-desktop/). Follow the instruction to install, and restart your machine if necessary

- Verify if the installation is good by opening up the Terminal (from MacOS), then type `docker`, you should see the following coming up if the installation is good


- build the docker image to use as the environment

```
docker build -t language-x-change-training:0.1 .

```

- to start the environment, run the following command

```
docker run -it --rm language-x-change-training:0.1 /bin/bash
```

- Logging to weights&biases (w&b) with the command line

```
wandb login
```

and you will be prompted to get the API key from the the given website, open the website and copy the key into the terminal

```
root@50d254b77e2b:/app# wandb login
wandb: Logging into wandb.ai. (Learn how to deploy a W&B server locally: https://wandb.me/wandb-server)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
wandb: Paste an API key from your profile and hit enter, or press ctrl+c to quit: 

...
...
wandb: Appending key for api.wandb.ai to your netrc file: /root/.netrc

root@50d254b77e2b:/app# 


```



- Logging to Hugging face with the command line

run the following command, and pasting in the personal token you created from hugging face.
```
huggingface-cli login

root@50d254b77e2b:/app# huggingface-cli login

    _|    _|  _|    _|    _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|_|_|_|    _|_|      _|_|_|  _|_|_|_|
    _|    _|  _|    _|  _|        _|          _|    _|_|    _|  _|            _|        _|    _|  _|        _|
    _|_|_|_|  _|    _|  _|  _|_|  _|  _|_|    _|    _|  _|  _|  _|  _|_|      _|_|_|    _|_|_|_|  _|        _|_|_|
    _|    _|  _|    _|  _|    _|  _|    _|    _|    _|    _|_|  _|    _|      _|        _|    _|  _|        _|
    _|    _|    _|_|      _|_|_|    _|_|_|  _|_|_|  _|      _|    _|_|_|      _|        _|    _|    _|_|_|  _|_|_|_|

    To login, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Token: 

...
...

Token has not been saved to git credential helper.
Your token has been saved to /root/.cache/huggingface/token
Login successful


```



- to run the training

```
python training.py
```

- to create sweeps in W&B

run `python create_sweep.py` and you will see the following output

```
root@50d254b77e2b:/app# python create_sweep.py 
Create sweep with ID: ngxfr1x4
Sweep URL: https://wandb.ai/poppysmic/language-x-change/sweeps/ngxfr1x4
root@50d254b77e2b:/app# 

```

(Assuming you have a different account name and the sweep ID will be different)


- to run have the agent run the sweep

run `wandb agent <YOUR_USERNAME>/language-x-change/<YOUR_SWEEP_ID>`

the agent will randomly select the different hyperparameters to run the experiment, and all data are plotted in the w&b dashboard under 'Sweeps'.

```
wandb: Starting wandb agent 🕵️
2024-01-25 20:10:54,496 - wandb.wandb_agent - INFO - Running runs: []
2024-01-25 20:10:54,786 - wandb.wandb_agent - INFO - Agent received command: run
2024-01-25 20:10:54,788 - wandb.wandb_agent - INFO - Agent starting run with config:
        learning_rate: 0.0001420854936531253
        num_train_epochs: 1
2024-01-25 20:10:54,793 - wandb.wandb_agent - INFO - About to run command: /usr/bin/env python training.py --learning_rate=0.0001420854936531253 --num_train_epochs=1
wandb: Currently logged in as: poppysmic. Use `wandb login --relogin` to force relogin
wandb: WARNING Ignored wandb.init() arg project when running a sweep.


...
...
...


```

Depending on the machine, it may take a while to run each round. but if you are willing to pay for the cloud instance to run the agent, this can be run in Azure / AWS cloud instance.