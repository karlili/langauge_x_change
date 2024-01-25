language-x-change Training

this docker image helps to encapsulate the complexity in the training process, by using weights & biases (W&B) sweeps

# Prerequisite 

- You will need to have Docker running on your machine. It is available for MacOS and Window machine. You can download a copy from here (https://www.docker.com/products/docker-desktop/). Follow the instruction to install, and restart your machine if necessary

- Verify if the installation is good by opening up the Terminal (from MacOS), then type `docker`, you should see the following coming up if the installation is good


Get ready your W&B authorized token and the huggingface token, then open up docker-compose.yml file.

 - To get the W&B token, go to https://wandb.ai/authorize, and copy the token from there
 - To get the huggingface token, go to https://huggingface.co/settings/token, and copy the token from there

Under environment, paste in the WANDB_API_KEY & HUGGING_FACE_HUB_TOKEN to the file respectivity

```
    environment:
      WANDB_API_KEY: "xxxxxxx"
      HUGGING_FACE_HUB_TOKEN: "yyyyyy"

```

Once the tokens are set, execute the following command instead, which will give the same environment
```
docker compose run training
```


- to run a single training experiment

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