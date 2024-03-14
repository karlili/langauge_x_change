
'''
 1.
 Before downloading any new dataset, 
 make sure to check if it needs to Check and Agrees to the terms first, otherwise the download would fail

 2.
 Before running the training script, make sre you have set the env variable for huggingface_hub
 export HF_HOME="/Volumes/DATA/huggingface/"

 3.
 If the training fails with this exception,
 'RuntimeError: MPS backend out of memory 
 (MPS allocated: 23.33 GB, other allocations: 5.32 GB, max allowed: 36.27 GB). Tried to allocate 7.93 GB on private pool.'
 export this variable before running the script, e.g.
 PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python training.py


'''
import wandb
import datetime

# from accelerate import Accelerator

now = datetime.datetime.now().strftime("%d-%m-%Y-%H%M")

config = {
    "model_name": "whisper-small-cantonese_" + now,
    
    "gradient_accumulation_steps": 4,  # increase by 2x for every 2x decrease in batch size
    "learning_rate": 2e-2,
    "warmup_steps": 100,
    "max_steps": 200,

    #gradient checkpointing and use_reentrant are related to each other  
    #if gradient checkpoint is True, set use_reentrant to True to potentially reducing memory usage
    "gradient_checkpointing": True,
    "use_reentrant": True,
    "use_cache":False,

    # evaluation strategy can be 'no', 'steps' or 'epoch'
    "evaluation_strategy": "steps", 

    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,

    "predict_with_generate": False,
    "generation_max_length": 50,


    
    "save_steps": 50,
    "eval_steps": 50,
    "logging_steps": 25,
    "metric_for_best_model": "wer",
    "num_train_epochs": 5,
    
}

wandb.init(project="language-x-change", config=config)

from datasets import load_dataset, DatasetDict

dataset_name = "mozilla-foundation/common_voice_16_0"
language_to_train = 'yue'

common_voice = DatasetDict()
common_voice["train"] = load_dataset(
    dataset_name, language_to_train,
    split="train+validation",
    trust_remote_code=True
    
)

common_voice["test"] = load_dataset(
    dataset_name, language_to_train,
    split="test",
 
)

# print(common_voice)


from transformers import WhisperFeatureExtractor

feature_extractor = WhisperFeatureExtractor.from_pretrained(
    "openai/whisper-small",
)  # start with the whisper small checkout

from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small",
                                             language="cantonese",
                                             task="transcribe",
                                             )

from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained("openai/whisper-small",
                                             language="cantonese",
                                             task="transcribe",
                                            #  cache_dir="/Volumes/BACKUP/Coding/HUGGING_FACE/processor"
                                             )

# Preparing Data

# Whisper expecting the audio to be at sampling rate @16000 - this is just to make sure the sampling rate fits whisper's training
# Since our input audio is sampled at 48kHz, we need to downsample it to 16kHz prior to passing it to the Whisper feature extractor, 
# 16kHz being the sampling rate expected by the Whisper model.
from datasets import Audio

raw_common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

# print(raw_common_voice["train"][0])


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


finalized_common_voice = raw_common_voice.map(prepare_dataset,
                                              remove_columns=raw_common_voice.column_names["train"],
                                              num_proc=2)
# print(finalized_common_voice)


import torch


from dataclasses import dataclass
from typing import Any, Dict, List, Union

# device = torch.device('mps')
device = torch.device('cpu')

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

import evaluate

metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    wandb.log({"wer": wer})
    return {"wer": wer}


from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small",
    # cache_dir="/Volumes/BACKUP/Coding/HUGGING_FACE/models"
)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(

    fp16=False,  # if we are not using CUDA or non graphics card, use fp16=false

    output_dir='model/'+wandb.config["model_name"],  # change to a repo name of your choice
    
    per_device_eval_batch_size=wandb.config["per_device_eval_batch_size"],
    per_device_train_batch_size=wandb.config["per_device_train_batch_size"],

    # generation_max_length=wandb.config["generation_max_length"],

    gradient_accumulation_steps=wandb.config["gradient_accumulation_steps"],
    
    learning_rate=wandb.config["learning_rate"],
    
    warmup_steps=wandb.config["warmup_steps"],
    max_steps=wandb.config["max_steps"],
    
    num_train_epochs=wandb.config["num_train_epochs"],

    gradient_checkpointing=wandb.config["gradient_checkpointing"],
    gradient_checkpointing_kwargs={
        "use_reentrant": wandb.config["use_reentrant"], # Set use_reentrant to True
        # "use_cache":wandb.config["use_cache"]
    },  

    evaluation_strategy=wandb.config["evaluation_strategy"],
    

    save_steps=wandb.config["save_steps"],
    eval_steps=wandb.config["eval_steps"],
    logging_steps=wandb.config["logging_steps"],

    report_to=["wandb"],  # this would requires the tensorboardx to be installed
    load_best_model_at_end=True,
    metric_for_best_model=wandb.config["metric_for_best_model"],
    
    greater_is_better=False,

    # push the model to huggingface hub
    push_to_hub=False,
)

from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=finalized_common_voice["train"],
    eval_dataset=finalized_common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)
trainer.train()

wandb.finish()
