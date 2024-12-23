"""
SentimentClassifier is a base class for all sentiment classifiers.
It provides a template for all sentiment classifiers to follow.
The methods in this class are abstract and must be implemented by the child class.
Some of the methods are:
    - train: trains the model
    - predict: predicts the sentiment of a given text
    - batch_predict: predicts the sentiment of a batch of texts
    - evaluate: evaluates the model on a given dataset
    - save_model: saves the model to a file
    - load_model: loads the model from a file
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

class SentimentClassifier(ABC):
    @abstractmethod
    def train(self, X: List[str], y: List[str], val_ratio) -> None:
        """
        Trains the model on the given data.

        Args:
        X: List[str]: A list of texts to train on.
        y: List[str]: A list of labels corresponding to the texts.
        """
        pass

    @abstractmethod
    def predict(self, X: str) -> str:
        """
        Predicts the sentiment of a given text.

        Args:
        X: str: The text to predict the sentiment of.

        Returns:
        str: The predicted sentiment of the text.
        """
        pass

    @abstractmethod
    def batch_predict(self, X: List[str]) -> List[str]:
        """
        Predicts the sentiment of a batch of texts.

        Args:
        X: List[str]: A list of texts to predict the sentiment of.

        Returns:
        List[str]: A list of predicted sentiments of the texts.
        """
        pass

    @abstractmethod
    def evaluate(self, X: List[str], y: List[str]) -> Tuple[float, float]:
        """
        Evaluates the model on the given data.

        Args:
        X: List[str]: A list of texts to evaluate the model on.
        y: List[str]: A list of labels corresponding to the texts.

        Returns:
        Tuple[float, float]: A tuple of the accuracy and f1 score of the model.
        """
        pass

    @abstractmethod
    def save_model(self, file_path: str) -> None:
        """
        Saves the model to a file.

        Args:
        file_path: str: The path to save the model to.
        """
        pass

    @abstractmethod
    def load_model(self, file_path: str) -> None:
        """
        Loads the model from a file.

        Args:
        file_path: str: The path to load the model from.
        """
        pass