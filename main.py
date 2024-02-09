from model import SentimentClassifier, RobertaClass
from utils import create_data_loader, train_epoch, eval_model
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    AutoTokenizer,
    BertTokenizer,
)
from collections import defaultdict

import pandas as pd
import pickle
import torch.nn as nn
import torch


if __name__ == "main":
    class_names = ["negativo", "neutro", "positivo"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "model_name"
    if model_name in [
        "neuralmind/bert-base-portuguese-cased",
        "bert-base-multilingual-cased",
    ]:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = SentimentClassifier(len(class_names)).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = RobertaClass(len(class_names)).to(device)

    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    df_val = pd.read_csv("val.csv")

    BATCH_SIZE = 16
    MAX_LEN = 300

    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

    EPOCHS = 4

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS

    warmup_proportion = 0.2

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=int(total_steps * warmup_proportion),
    )

    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0
    for epoch in range(EPOCHS):

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train),
        )

        print(f"Train loss {train_loss} accuracy {train_acc}")

        val_acc, val_loss = eval_model(
            model, val_data_loader, loss_fn, device, len(df_val)
        )

        print(f"Val   loss {val_loss} accuracy {val_acc}")
        print()

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        if val_acc > best_accuracy:
            with open("bertimbau_model.pkl", "wb") as file:
                pickle.dump(model, file)

            with open("history.pkl", "wb") as file:
                pickle.dump(history, file)

            best_accuracy = val_acc
