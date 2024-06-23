#Libraries
import sys
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import os
import json
import pandas as pd
import torch
import re
import numpy as np
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from tokenizers import AddedToken
from transformers import (DataCollatorForSeq2Seq, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, TextDataset, AutoConfig)
from rouge_score import rouge_scorer

import nltk
nltk.download('punkt')

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_data_from_json(data_dir, file_name):
    return [json.loads(line) for line in open(os.path.join(data_dir, file_name), "r")]

def create_combined_json(data_dir):
    #Train
    train_data = []
    train_data.extend(get_data_from_json(data_dir, "PLOS_train.jsonl"))
    train_data.extend(get_data_from_json(data_dir, "eLife_train.jsonl"))
    with open("combined_train.jsonl", "w") as train_file:
        for item in train_data:
            json.dump(item, train_file)
            train_file.write('\n')

    #Val
    val_data = []
    val_data.extend(get_data_from_json(data_dir, "PLOS_val.jsonl"))
    val_data.extend(get_data_from_json(data_dir, "eLife_val.jsonl"))
    with open("combined_val.jsonl", "w") as val_file:
        for item in val_data:
            json.dump(item, val_file)
            val_file.write('\n')

    print("combined files created and stored!")

def calc_rouge(preds, refs):
    # Get ROUGE F1 scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], \
                                    use_stemmer=True, split_summaries=True)
    scores = [scorer.score(p, refs[i]) for i, p in enumerate(preds)]
    return np.mean([s['rouge1'].fmeasure for s in scores]), \
            np.mean([s['rouge2'].fmeasure for s in scores]), \
            np.mean([s['rougeLsum'].fmeasure for s in scores])

def compute_metrics(pred, tokenizer, batched=True):

    # Extract the label IDs and predicted IDs from the input NamedTuple
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # Decode the predicted and label IDs to strings, skipping special tokens
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # Round the Rouge2 scores to 4 decimal places and return them in a dictionary
    scores = calc_rouge(pred_str, label_str)
    return {
        "rouge1": scores[0],
        "rouge2": scores[1],
        "rougeLsum": scores[2] 
    }

def preprocess_text(text):
    pattern = r'\([^()]*\)|\{[^{}]*\}|\[[^\[\]]*\]'
    text = re.sub(pattern, '', text)
    return text
    
def tokenize_text(text, max_input_length, tokenizer):
    text = preprocess_text(text)
    pattern = r'\.(?=\s+[A-Z])'
    lines = re.split(pattern, text)
    total_tokens = 0
    encoded_ids = tokenizer("summarize this article: ")['input_ids'][:-1]
    i = 0
    #only the first 500 words are taken into account
    while len(encoded_ids) < 500 and i < len(lines):
        lines[i] += "."
        input_ids = tokenizer(lines[i])['input_ids'][:-1]
        if (len(encoded_ids) + len(input_ids) < 500):
            encoded_ids += input_ids
            i += 1
        else:
            break
    input_ids = tokenizer("summary: ")['input_ids']
    encoded_ids += input_ids

    #Till here, maximum possible lines are included inside encoded_ids

    #Padding explicitly handled
    
    attention_mask = [1]*len(encoded_ids) + [0]*(max_input_length - len(encoded_ids))
    encoded_ids += [tokenizer.pad_token_id]*(max_input_length - len(encoded_ids))
    
    output = {"input_ids": encoded_ids, "attention_mask": attention_mask}
    return output

def load_article_dataset(filepath):
    df = pd.read_json(filepath, lines=True)
    return Dataset.from_pandas(df)

def process_data_to_model_inputs(
    batch, tokenizer, max_input_length, max_output_length
):
    # tokenize the inputs and outputs using the provided tokenizer
    inputs = {} 
    input_ids_list = []
    attention_mask_list = []
    for article in batch['article']:
        tokenized_output = tokenize_text(article, max_input_length, tokenizer)
        input_ids_list.append(tokenized_output['input_ids'])
        attention_mask_list.append(tokenized_output['attention_mask'])
    input_ids = torch.tensor(input_ids_list)
    
    attention_mask = torch.tensor(attention_mask_list)
    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    outputs = tokenizer(
        batch["lay_summary"],
        padding="max_length",
        truncation=True,
        max_length=max_output_length,
    )

    # create a dictionary to store the preprocessed model inputs
    batch["input_ids"] = inputs['input_ids']
    batch["attention_mask"] = inputs['attention_mask'] 
    batch["labels"] = outputs.input_ids
    batch["labels"] = [
        [
            -100 if token == tokenizer.pad_token_id else token
            for token in labels
        ]
        for labels in batch["labels"]
    ]
    return batch 

def create_article_dataset_dict(
    filename, # dict {"train": train_path, "val": val_path}
    batch_size,
    tokenizer,
    max_input_length,
    max_output_length
):
    dataset_types = ["train", "val"]
    datasets = {}
    
    for dtype in dataset_types:
        # load the dataset 
        dataset = load_article_dataset(filename[dtype])
        
        dataset = dataset.map(
            process_data_to_model_inputs,
            batched=True,
            batch_size=batch_size,
            remove_columns=["article", "lay_summary"],
            fn_kwargs = {
                "tokenizer": tokenizer,
                "max_input_length": max_input_length,
                "max_output_length": max_output_length,
            },
        )

        dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "labels"
            ]
        )
        datasets[dtype] = dataset
    return DatasetDict(datasets)

def test_file(path_to_data, path_to_model, path_to_result, mode):
    if mode == "plos":
        input_file = "PLOS_test.jsonl"
        output_file = "plos.txt"
    elif mode == "elife":
        input_file = "eLife_test.jsonl"
        output_file = "elife.txt"

    test_path = os.path.join(path_to_data, input_file)
    store_path = os.path.join(path_to_result, output_file)

    MODEL_NAME = path_to_model

    generation_config = GenerationConfig.from_pretrained(MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map="auto").to(device)

    data_lst = [json.loads(line) for line in open(test_path, "r")]

    new_data_lst = []

    for data in tqdm(data_lst):
        input_packet = tokenize_text(data['article'], 512, tokenizer)
        input_ids = torch.tensor([input_packet['input_ids']]).to(device)
        outputs = model.generate(input_ids, max_new_tokens = 600, num_beams = 3, generation_config=generation_config)
        new_data_lst.append(tokenizer.decode(outputs[0], skip_special_tokens=True).replace("\n",""))

    with open(store_path, 'w') as outfile:
        for text in new_data_lst:
            outfile.write(text + "\n")


def train(path_to_data, path_to_save):
    print("[MS]Training ...")
    # print("path to data: ", path_to_data)
    # print("path o save: ", path_to_save)

    create_combined_json(path_to_data)

    train_path = "combined_train.jsonl"
    val_path = "combined_val.jsonl"

    #HYPERPARAMETERS ###############################################################
    MODEL_NAME = "google/flan-t5-base"
    
    dataloader_batch_size = 128 
    
    max_input_length = 512
    max_output_length = 300

    training_args = Seq2SeqTrainingArguments(
        output_dir=path_to_save,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=16,
        learning_rate=1e-3,
        predict_with_generate=True,
        num_train_epochs=25,
        logging_steps=200,
        save_steps=200,
        warmup_steps=300,
        evaluation_strategy="steps",
        save_total_limit=1,
        fp16=False,
        gradient_accumulation_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="rougeLsum",
        lr_scheduler_type="cosine"
    )

    ##################################################################################

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, legacy=False)

    filename = {'train': train_path, 'val': val_path}

    
    dataset_dict = create_article_dataset_dict(
        filename,
        dataloader_batch_size,
        tokenizer,
        max_input_length,
        max_output_length
    )
    
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.max_length = 512
    config.min_length = 256
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, config=config)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics= lambda x: compute_metrics(x, tokenizer, batched=True),  # The `compute_metrics` function needs to be defined before calling this.
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['val'],
        data_collator=data_collator
    )

    trainer.train()
    print("[MS]Training complete!")

    trainer.save_model(output_dir=path_to_save)
    print("[MS]Final model saved!")
    

def test(path_to_data, path_to_model, path_to_result):
    print("[MS]Testing ...")
    # print("path to data: ", path_to_data)
    # print("path to model: ", path_to_model)
    # print("path to result: ", path_to_result)
    prospective_folders = []
    for l in os.listdir(path_to_model):
        if "checkpoint" in l:
            prospective_folders += [os.path.join(path_to_model, l)]

    new_path_to_model = prospective_folders[0]

    print("[MS]Testing PLOS ..")
    test_file(path_to_data, new_path_to_model, path_to_result, mode="plos")
    print("[MS]Testing PLOS complete!")

    print("[MS]Testing eLife ..")
    test_file(path_to_data, new_path_to_model, path_to_result, mode="elife")
    print("[MS]Testing eLife complete!")

def main():
    mode = sys.argv[1]
    if mode == "train":
        PATH_TO_DATA = sys.argv[2]
        PATH_TO_SAVE = sys.argv[3]
        
        train(path_to_data=PATH_TO_DATA, path_to_save=PATH_TO_SAVE)

    elif mode == "test":
        PATH_TO_DATA = sys.argv[2]
        PATH_TO_MODEL = sys.argv[3]
        PATH_TO_RESULT = sys.argv[4]
    
        test(path_to_data=PATH_TO_DATA, path_to_model=PATH_TO_MODEL, path_to_result=PATH_TO_RESULT)

    else:
        print("Mode could either be train or test!")


if __name__ == "__main__":
    main()