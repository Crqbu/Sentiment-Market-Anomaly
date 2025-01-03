import json
import hydra
from omegaconf import DictConfig
from Models.BERTSent import BERTSent  # BERTSent.py in models class
from Models.LSTMSent import LSTMSent  # LSTMSent.py in models class

def load_data(data_path):
    with open(data_path, 'r') as file:
        data = json.load(file)
    texts = [entry['text'] for entry in data]
    labels = [entry['label'] for entry in data]
    return texts, labels

@hydra.main(config_path="Config", config_name="config")
def main(cfg: DictConfig):
    print(f"Using model configuration: {cfg.model.type}")
    # Load training data
    texts, labels = load_data(cfg.train_data_path)

    # Initialize the model based on configuration
    if cfg.model.typ.startswith('bert'):
        classifier = BERTSent.BERTSent(cfg)
    elif cfg.model.type.startswith('lstm'):
        classifier = LSTMSent.LSTMSent(cfg)
    else:
        raise ValueError("Unsupported model type specified in the configuration.")

    # Train the model
    classifier.train(texts, labels)
    
    # Optionally, save the model
    classifier.save_model('path_to_save_model')  # You need to implement this method in BERTSent and LSTMSent if not done

if __name__ == "__main__":
    main()
