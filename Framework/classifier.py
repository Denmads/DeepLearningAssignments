from typing import Protocol


class Classifier(Protocol):

    def train(self, samples, labels):
        """Trains the model using the passed arguments"""

    def predict(self, samples) -> object:
        """Predicts the labels for the given samples"""