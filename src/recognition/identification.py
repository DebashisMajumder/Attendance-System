import cv2
from dataclasses import dataclass

@dataclass
class FaceIdentifier:
    def __init__(self, detector, aligner, embedder, classifier):
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.classifier = classifier
        
    def identify(self, frame):
        image = cv2.imread(frame)
        
        if image is None:
            raise ValueError(f"Could not read the image from the path: {frame}")
        
        faces = self.detector.detect(frame)
        
        results = []
        
        for face in faces:
            bbox = face['bbox']
            face_crop = self.aligner.crop(image, bbox)
            
            if face_crop is None or face_crop.size == 0:
                continue
            
            embedding = self.embedder.generate(face_crop)
            
            roll_number, distance, winning_label = self.classifier.predict(embedding)
            
            results.append({
                "roll_number": roll_number if roll_number is not None else "Unknown",
                "bbox": bbox,
                "detection_confidence": face["confidence"],
                "recognition_distance": distance,
                "winning_label": winning_label
            })
        
        return results