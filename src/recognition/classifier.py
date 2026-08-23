from collections import Counter
import joblib
import numpy as np
from dataclasses import dataclass
 
@dataclass
class FaceClassifier:
    def __init__(self, model_path, k_neighbors=5, distance_threshold=0.5):
        self.model = joblib.load(model_path)
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
 
    def predict(self, embedding):
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
 
        distances, indices = self.model.kneighbors(embedding, n_neighbors=self.k_neighbors)
        
        distances = distances[0]
        indices = indices[0]
        encoded_labels = self.model._y[indices]
        neighbor_labels = self.model.classes_[encoded_labels]
        winning_label, _ = Counter(neighbor_labels).most_common(1)[0]
        
        same_label_distances = distances[neighbor_labels == winning_label]
        confidence_distance = float(np.mean(same_label_distances))
 
        if confidence_distance > self.distance_threshold:
            return None, winning_label, confidence_distance
 
        return winning_label, confidence_distance, winning_label