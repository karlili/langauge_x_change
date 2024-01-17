import wandb


sweep_config = {
    'method': 'random',
    'program': 'training.py',
    'name': 'sweep-whisper-finetune',
    'project': 'language-x-change',

}

metric = {
    'name': 'wer',
    'goal': 'minimize'
}

parameters_dict = {
    'learning_rate': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.001
        },

    }
parameters_dict.update({
    'num_train_epochs': {
        'value': 1
    }
})

sweep_config['metric'] = metric
sweep_config['parameters'] = parameters_dict

sweep_id = wandb.sweep(sweep_config, project="language-x-change")