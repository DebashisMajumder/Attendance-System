import numpy as np
import insightface
from dataclasses import dataclass

@dataclass
class FaceEmbedder:
    def __init__(self, model_path, ctx_id = 0):
        self.model = insightface.model_zoo.get_model(model_path, ctx_id=ctx_id)
        self.model.prepare(ctx_id=ctx_id)
        
    def generate(self, face_image):
            embedding = self.model.get_feat(face_image)
            embedding = embedding.flatten()
            
            # L2 Normalization
            norm = np.linalg.norm(embedding)
            if norm == 0:
                raise ValueError("Invalid embedding: Norm is zero.")
            
            embedding = embedding / norm
            
            return embedding