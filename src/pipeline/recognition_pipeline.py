from collections import Counter
from dataclasses import dataclass

@dataclass
class RecognitionPipeline:

     def __init__(self, identifier, min_detection_confidence=0.5, min_agreement=3):
        self.identifier = identifier
        self.min_detection_confidence = min_detection_confidence
        self.min_agreement = min_agreement
        
     def process(self, frame_paths):

        predictions = []

        for frame_path in frame_paths:

            results = self.identifier.identify(
                frame_path
            )

            for result in results:
                confidence = result["detection_confidence"]
                if confidence is None or confidence <= self.min_detection_confidence:
                    continue
                
                predictions.append(result["roll_number"])
                print(f"  {frame_path} -> {result['roll_number']} "f"(distance: {result['recognition_distance']:.4f}," f"winning label: {result['winning_label']}")
                    
        if not predictions:
            return {
                "status": "unknown",
                "roll_number": None,
                "votes": 0,
                "total_predictions": 0
            }

        # Majority voting
        counter = Counter(predictions)

        winner, votes = counter.most_common(1)[0]

        if winner != "Unknown" and votes >= self.min_agreement:
            return {
                "status": "recognized",
                "roll_number": winner,
                "votes": votes,
                "total_predictions": len(predictions),
            }
 
        return {
            "status": "invalid",
            "roll_number": None,
            "votes": votes,
            "total_predictions": len(predictions),
        }
