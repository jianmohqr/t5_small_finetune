
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os


from transformers import T5Tokenizer, T5ForConditionalGeneration

from torch import cuda
device = 'cuda' #if cuda.is_available() else 'cpu'


import pandas as pd

path = "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"
df = pd.read_csv(path)

df.head()




from rich.table import Column, Table
from rich import box
from rich.console import Console


console = Console(record=True)



def display_df(df):
    """display dataframe in ASCII format"""

    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], row[1])

    console.print(table)



training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)






class YourDataSetClass(Dataset):

    def __init__(
        self, dataframe, tokenizer, source_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_ids_y": target_ids.to(dtype=torch.long),
        }




def train(epoch, tokenizer, model, device, loader, optimizer):

    model.train()
    for _, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)

        outputs = model(
            input_ids=ids,
            attention_mask=mask,
            decoder_input_ids=y_ids,
            labels=lm_labels,
        )
        loss = outputs[0]

        if _ % 10 == 0:
            training_logger.add_row(str(epoch), str(_), str(loss))
            console.print(training_logger)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



def validate(epoch, tokenizer, model, device, loader):

  model.eval()
  predictions = []
  actuals = []
  loss = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):
          y = data['target_ids'].to(device, dtype = torch.long)
          ids = data['source_ids'].to(device, dtype = torch.long)
          mask = data['source_mask'].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]

        
          label_id = y.clone().detach().to(device, dtype=torch.long)
          label_id[y == tokenizer.pad_token_id] = -100
          predict_loss = model(
            input_ids = ids,
            attention_mask = mask,
            labels = label_id
            )['loss']

          if _%10==0:
              console.print(f'Completed {_}')
              console.print(f'Loss {predict_loss}')

          predictions.extend(preds)
          loss.extend(predict_loss)
          actuals.extend(target)
  return predictions, loss, actuals





def T5Trainer(dataframe, source_text, target_text, model_params, output_dir="./outputs/"):


    torch.manual_seed(model_params["SEED"])  
    np.random.seed(model_params["SEED"]) 
    torch.backends.cudnn.deterministic = True


    console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")


    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])


    # model = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
    model = T5ForConditionalGeneration.from_pretrained('outputs/model_files')
    model = model.to(device)


    console.log(f"[Data]: Reading data...\n")

 
    dataframe = dataframe[[source_text, target_text]]
    display_df(dataframe.head(2))


    train_size = 0.8
    train_dataset = dataframe.sample(frac=train_size, random_state=model_params["SEED"])
    val_dataset = dataframe.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    console.print(f"FULL Dataset: {dataframe.shape}")
    console.print(f"TRAIN Dataset: {train_dataset.shape}")
    console.print(f"TEST Dataset: {val_dataset.shape}\n")

    
    training_set = YourDataSetClass(
        train_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )
    val_set = YourDataSetClass(
        val_dataset,
        tokenizer,
        model_params["MAX_SOURCE_TEXT_LENGTH"],
        model_params["MAX_TARGET_TEXT_LENGTH"],
        source_text,
        target_text,
    )

    
    train_params = {
        "batch_size": model_params["TRAIN_BATCH_SIZE"],
        "shuffle": True,
        "num_workers": 0,
    }

    val_params = {
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "shuffle": False,
        "num_workers": 0,
    }

    
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)


    # Freeze all the attention layers
    modules_to_freeze = [model.encoder.block[i].layer[0] for i in range(len(model.encoder.block))]
    modules_to_freeze.extend([model.decoder.block[i].layer[0] for i in range(len(model.decoder.block))])
    modules_to_freeze.extend([model.decoder.block[i].layer[1] for i in range(len(model.decoder.block))])

    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False


    # Define the optimizer
    optimizer = torch.optim.Adam(
        params = filter(lambda p: p.requires_grad, model.parameters()), lr = model_params["LEARNING_RATE"]
    )

    # Training 
    console.log(f"[Initiating Fine Tuning]...\n")

    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, device, training_loader, optimizer)

    console.log(f"[Saving Model]...\n")
    
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


    # Validation
    console.log(f"[Initiating Validation]...\n")
    for epoch in range(model_params["VAL_EPOCHS"]):

        predictions, loss, actuals = validate(epoch, tokenizer, model, device, val_loader)

        model_finetune1 = T5ForConditionalGeneration.from_pretrained('outputs/t5_small_finetune_model_files')
        model_finetune1 = model_finetune1.to(device)
        predictions_f1, loss_f1 = validate(epoch, tokenizer, model_finetune1, device, val_loader)[0:2]

        model_original = T5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
        model_original = model_original.to(device)
        predictions_org, loss_org = validate(epoch, tokenizer, model_original, device, val_loader)[0:2]

        
        final_df = pd.DataFrame({
            "Loss": loss,
            "Loss_finetune1": loss_f1,
            "Loss_original_t5": loss_org, 
            "Generated Text (with freeze)": predictions, 
            "Generated Text (finetune1)": predictions_finetune1, 
            "Generated Text (original)": predictions_original,
            "Ground Truth": actuals
            })
        final_df = pd.DataFrame({
            "Loss": loss,
            "Loss_finetune1": loss_f1,
            "Loss_original_t5": loss_org, 
            "Generated Text (with freeze)": predictions, 
            "Generated Text (finetune1)": predictions_finetune1, 
            "Generated Text (original)": predictions_original,
            "Ground Truth": actuals
            })
        final_df.to_csv(os.path.join(output_dir, "predictions.csv"))

    console.save_text(os.path.join(output_dir, "logs.txt"))

    console.log(f"[Validation Completed.]\n")
    console.print(
        f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n"""
    )
    console.print(
        f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n"""
    )
    console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")





model_params = {
    "MODEL": "t5-small",  # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE": 8,  # training batch size
    "VALID_BATCH_SIZE": 128,  # validation batch size
    "TRAIN_EPOCHS": 3,  # number of training epochs
    "VAL_EPOCHS": 1,  # number of validation epochs
    "LEARNING_RATE": 1e-4,  # learning rate
    "MAX_SOURCE_TEXT_LENGTH": 512,  # max length of source text
    "MAX_TARGET_TEXT_LENGTH": 50,  # max length of target text
    "SEED": 23,  # set seed for reproducibility
}



df["text"] = "summarize: " + df["text"]

T5Trainer(
    dataframe=df,
    source_text="text",
    target_text="headlines",
    model_params=model_params,
    output_dir="outputs",
)





'''
evaluation_logger = Table(
    Column("Loss", justify="center"),
    Column("Loss_f1", justify="center"),
    Column("Loss_org", justify="center"),
    Column("Prediction", justify="center"),
    Column("Ground Truth", justify="center"),
    title="Loss and prediction",
    pad_edge=False,
    box=box.ASCII,
)



## Loading the fine-tuned model
from transformers import T5Tokenizer, T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained('outputs/model_files/')
model_finetune1 = T5ForConditionalGeneration.from_pretrained('outputs/t5_small_finetune_model_files/')
model_original = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

import pandas as pd

path = "https://raw.githubusercontent.com/Shivanandroy/T5-Finetuning-PyTorch/main/data/news_summary.csv"
df = pd.read_csv(path)

df.head()

evaluation_dataset = val_dataset
evaluation_dataset.head()

for i in range(100):
    try_in = 'summarize: ' + evaluation_dataset['text'][i]
    try_out = evaluation_dataset['headlines'][i]

    in_info = tokenizer.encode_plus(
        try_in, max_length=model_params['MAX_SOURCE_TEXT_LENGTH'], 
        truncation=True, padding="max_length", return_tensors="pt"
        )
    out_info = tokenizer.encode_plus(
        try_out, max_length=model_params['MAX_TARGET_TEXT_LENGTH'], 
        truncation=True, padding="max_length", return_tensors="pt"
        )
    labels = out_info['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100

    loss = model(input_ids=in_info['input_ids'], attention_mask=in_info['attention_mask'], labels=out_info['input_ids'])['loss']
    loss_f1 = model_finetune1(input_ids=in_info['input_ids'], attention_mask=in_info['attention_mask'], labels=out_info['input_ids'])['loss']
    loss_org = model_original(input_ids=in_info['input_ids'], attention_mask=in_info['attention_mask'], labels=out_info['input_ids'])['loss']
    # console.print(loss) 

    generated_id = model.generate(input_ids = in_info['input_ids'], attention_mask = in_info['attention_mask'])[0]
    generated = tokenizer.decode(generated_id, skip_special_tokens = True, clean_up_tokenization_spaces = True)
    #console.print(generated, try_out)
    evaluation_logger.add_row(str(loss), str(loss_f1), str(loss_org), generated, try_out)
    console.print(evaluation_logger)

'''