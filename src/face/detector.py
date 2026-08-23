from retinaface import RetinaFace
from dataclasses import dataclass

@dataclass
class FaceDetector:
    def detect(self, frame):
        faces = RetinaFace.detect_faces(frame)
        
        if not faces:
            print("No faces detected!")
            return[]
        
        results = []
        
        for _, face_data in faces.items():
            x1, y1, x2, y2 = face_data["facial_area"]
            
            results.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": face_data.get(
                "score", None),
            })
            
        
        return results