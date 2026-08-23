import cv2
from dataclasses import dataclass

@dataclass
class FaceAligner:
    def __init__(self, padding_ratio = 0.2):
        self.padding_ratio = padding_ratio
        
    def crop(self, image, bbox):
        x1, y1, x2, y2 = bbox
        
        width = x2 - x1
        height = y2 - y1
        
        padding_x = int(width * self.padding_ratio)
        padding_y = int(height * self.padding_ratio)
        
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(image.shape[1], x2 + padding_x)
        y2 = min(image.shape[0], y2 + padding_y)
        
        face = image[y1:y2, x1:x2]
        
        return face