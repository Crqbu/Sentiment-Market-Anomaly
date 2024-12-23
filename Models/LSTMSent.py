import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import List
from tqdm import tqdm
from datetime import datetime
import hydra
from omegaconf import DictConfig

# Custom LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, bidirectional, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        direction_factor = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * direction_factor, num_classes)

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_out, _ = self.lstm(embedded)
        # Take the output from the last LSTM cell
        final_feature_map = self.dropout(lstm_out[:, -1, :])
        final_out = self.fc(final_feature_map)
        return final_out

class LSTMSent:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMClassifier(
            vocab_size=config.model.vocab_size,
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_classes=config.model.num_classes,
            bidirectional=config.model.bidirectional,
            dropout=config.model.dropout
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.weight_decay)

    def train(self, X: List[int], y: List[int], val_ratio=0.1):
        # Prepare dataset
        X_tensor = torch.tensor(X, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        train_size = int((1 - val_ratio) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=self.config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.evaluation.eval_batch_size, shuffle=False)

        checkpoint_dir = self.config.training.checkpoint_dir

        # Training loop
        for epoch in range(self.config.training.epoch):
            self.model.train()
            total_loss = 0
            for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            print(f"Epoch {epoch + 1}, Average Training Loss: {total_loss / len(train_loader)}")

            # Validation
            self.model.eval()
            total_eval_loss = 0
            correct = 0
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc="Validating"):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    total_eval_loss += loss.item()
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
            
            print(f"Validation Loss: {total_eval_loss / len(val_loader)}, Accuracy: {correct / len(val_dataset)}")

            # Save checkpoint
            checkpoint_path = f"{checkpoint_dir}/lstm_checkpoint_date_{datetime.now()}_epoch_{epoch + 1}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)

    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            text_tensor = torch.tensor(text).to(self.device)
            output = self.model(text_tensor)
            _, predicted = torch.max(output, 1)
        return predicted.item()

    def batch_predict(self, texts):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for text in texts:
                text_tensor = torch.tensor(text).to(self.device)
                output = self.model(text_tensor)
                _, predicted = torch.max(output, 1)
                predictions.append(predicted.item())
        return predictions

    def evaluate(self, X, y):
        self.model.eval()
        eval_dataset = DataLoader(list(zip(X, y)), batch_size=self.config.evaluation.eval_batch_size)
        total_accuracy = 0
        total_loss = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        with torch.no_grad():
            for inputs, labels in eval_dataset:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_accuracy += (predicted == labels).sum().item()
                total_tp += ((predicted == 1) & (labels == 1)).sum().item()
                total_fp += ((predicted == 1) & (labels == 0)).sum().item()
                total_fn += ((predicted == 0) & (labels == 1)).sum().item()
        precision = total_tp / (total_tp + total_fp)
        recall = total_tp / (total_tp + total_fn)
        f1_score = 2 * (precision * recall) / (precision + recall)
        return total_loss / len(eval_dataset), total_accuracy / len(eval_dataset.dataset), f1_score

    def save_model(self, file_path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, file_path)

    def load_model(self, file_path):
        checkpoint = torch.load(file_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        self.model.to(self.device)
    

@hydra.main(config_path="Config", config_name="config")
def main(cfg: DictConfig):
    lstm_classifier = LSTMSent(cfg)
    # Assuming X and y are loaded appropriately
    # lstm_classifier.train(X, y)

if __name__ == "__main__":
    main()
