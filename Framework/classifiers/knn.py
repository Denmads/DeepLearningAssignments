from external.k_nearest_neighbor import KNearestNeighbor

class KnnClassifier:
    def __init__(self):
        self.internal = KNearestNeighbor()

    def train(self, samples, labels):
        """Trains the model using the passed arguments"""
        self.internal.train(samples, labels)

    def predict(self, samples, k, num_loops=0) -> object:
        """Predicts the labels for the given samples"""
        return self.internal.predict(samples, k, num_loops)
