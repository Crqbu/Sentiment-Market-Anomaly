"""
BERTSent is a model that uses ahmedrachid/FinancialBERT-Sentiment-Analysis model from huggingface to predict the sentiment of a given text.
it inherits from SentimentClassifier and implements the abstract methods.
We also use OmegaConf and hydra to manage configurations of the model.
The configs are stored in the config.yaml file in the Config directory.
Config directory contains folders:
- dataset: contains the dataset configurations
- model: contains the model configurations
- optimizer: contains the optimizer configurations
- scheduler: contains the scheduler configurations
The configs for the BERT variant are stored under bert.yaml in each of the folders.
make sure to use hydra to initialize the model with the configurations.
"""
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader, random_split
import hydra
from datetime import datetime
import os

class BERTSent:
    def __init__(self, config):
        self.config = config
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("mps")
        self.model = self.initialize_model()
        self.tokenizer = self.initialize_tokenizer()

    def initialize_model(self):
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.bert_model,
            num_labels=self.config.num_labels,
            hidden_dropout_prob=self.config.hidden_dropout_prob,
            attention_probs_dropout_prob=self.config.attention_probs_dropout_prob
        )
        model.to(self.device)
        return model

    def initialize_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
        return tokenizer

    def predict(self, X: str) -> str:
        self.model.eval()
        inputs = self.tokenizer(X, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_seq_length)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction = outputs.logits.argmax(-1).item()
        return prediction

    def batch_predict(self, X: List[str]) -> List[int]:
        self.model.eval()  # Ensure the model is in evaluation mode
        predictions = []

        # Choose a batch size
        batch_size = 16  # Adjust based on your GPU/CPU memory availability

        # Process data in batches
        for i in tqdm(range(0, len(X), batch_size), desc="Predicting"):
            batch_texts = X[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_seq_length)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
            logits = outputs.logits.argmax(-1).cpu().numpy()

            # Adjust predictions based on configuration
            if self.config.type in ["bert_mrm8488", "bert_cardiff", "bert_nickmuchi"]:
                logits = [pred - 1 for pred in logits]
            elif self.config.type == "bert_ahmedrachid":
                mapping = {0: -1, 1: 1, 2: 0}
                logits = [mapping[pred] for pred in logits]

            predictions.extend(logits)

        return predictions
    

    def train(self, X: List[str], y: List[int], val_ratio=0.1) -> None:
        # Ensure the checkpoint directory exists
        checkpoint_dir = self.config.training.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Prepare the dataset
        inputs = self.tokenizer(X, return_tensors="pt", padding=True, truncation=True, max_length=self.config.model.max_seq_length)
        labels = torch.tensor(y)
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
        
        # Split dataset into training and validation sets
        train_size = int((1 - val_ratio) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.training.eval_batch_size, shuffle=False)

        # Optimization setup
        optimizer = AdamW(self.model.parameters(), lr=self.config.training.learning_rate)
        total_steps = len(train_loader) * self.config.training.epoch
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * self.config.training.warmup_proportion), num_training_steps=total_steps)

        # Training loop
        for epoch in range(self.config.training.epoch):
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.training.epoch} Training"):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                self.model.zero_grad()
                outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            print(f"Average Training Loss: {avg_train_loss:.2f}")

            # Validation loop
            self.model.eval()
            val_loss, val_accuracy = 0, 0
            for batch in tqdm(val_loader, desc="Validation"):
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs.loss
                    logits = outputs.logits
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1).flatten()
                val_accuracy += (preds == b_labels).cpu().numpy().mean()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_accuracy / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.2f}, Accuracy: {avg_val_accuracy:.2f}")

            # Save checkpoint after each epoch
            checkpoint_path = os.path.join(checkpoint_dir, f"bert_checkpoint_date_{datetime.now()}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
                'validation_loss': avg_val_loss,
                'validation_accuracy': avg_val_accuracy
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")


    def evaluate(self, X: List[str], y: List[int]) -> Tuple[float, float, float]:
        predictions = self.batch_predict(X)
        accuracy = np.mean([pred == label for pred, label in zip(predictions, y)])
        f1_score = f1_score(y, predictions, average='weighted')
        loss = self.compute_loss(X, y)
        return loss, accuracy, f1_score

    def save_model(self, file_path: str) -> None:
        self.model.save_pretrained(file_path)
        self.tokenizer.save_pretrained(file_path)

    def load_model(self, file_path: str) -> None:
        self.model = AutoModelForSequenceClassification.from_pretrained(file_path)
        self.tokenizer = AutoTokenizer.from_pretrained(file_path)
        self.model.to(self.device)


@hydra.main(config_path="Config", config_name="config")
def main(cfg):
    bert_classifier = BERTSent(cfg)
    # Assume dataset is loaded here with `texts` and `labels`
    # accuracy = bert_classifier.evaluate(texts, labels)
    # print(f"Model accuracy: {accuracy}")

if __name__ == "__main__":
    main()
