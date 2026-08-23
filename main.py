from pathlib import Path

from src.face.detector import FaceDetector
from src.face.aligner import FaceAligner
from src.face.embedder import FaceEmbedder
from src.recognition.classifier import FaceClassifier
from src.recognition.identification import FaceIdentifier
from src.pipeline.recognition_pipeline import RecognitionPipeline

#-----------------------------------------------------------------------Path
CAM_FRAME_PATH =  Path("/home/debashis/SIH'26/Inter Clg/Attendance-System/cam_frames")
EMBEDDING_MODEL_PATH = Path("/home/debashis/SIH'26/Inter Clg/Attendance-System/models/w600k_r50.onnx")
TRAINED_CLASSIFIER_MODEL_PATH = Path("/home/debashis/SIH'26/Inter Clg/Attendance-System/models/knn_model.pkl")

#-----------------------------------------------------------------------Config
EXPECTED_FRAMES = 5

GPU_CTX_ID = 0

#---------------------------------------------------------------------Checking required files
if not EMBEDDING_MODEL_PATH.exists():
    raise FileNotFoundError(f"Embedding model file not found at {EMBEDDING_MODEL_PATH}")

if not TRAINED_CLASSIFIER_MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained classifier model file not found at {TRAINED_CLASSIFIER_MODEL_PATH}")

if not CAM_FRAME_PATH.exists():
    raise FileNotFoundError(f"Camera frames directory not found at {CAM_FRAME_PATH}")

#---------------------------------------------------------------------Initialize components
detector = FaceDetector()
aligner = FaceAligner(padding_ratio=0.2)
embedder = FaceEmbedder(model_path=str(EMBEDDING_MODEL_PATH), ctx_id = GPU_CTX_ID)
classifier = FaceClassifier(model_path=str(TRAINED_CLASSIFIER_MODEL_PATH))
identifier = FaceIdentifier(detector=detector, aligner=aligner, embedder=embedder, classifier=classifier)
pipeline = RecognitionPipeline(identifier=identifier, min_detection_confidence=0.5,)
frames = sorted(CAM_FRAME_PATH.glob("*.jpeg")) + sorted(CAM_FRAME_PATH.glob("*.jpg")) + sorted(CAM_FRAME_PATH.glob("*.png"))

#---------------------------------------------------------------------Process frames
# if len(frames) < EXPECTED_FRAMES:
#     print(f"Not enough frames for processing. Expected {EXPECTED_FRAMES}, but found {len(frames)}.")
#     raise SystemExit(1)
# if len(frames) > EXPECTED_FRAMES:
#     frames = frames[-EXPECTED_FRAMES:] ## min_agreement=EXPECTED_FRAMES
    
results = pipeline.process([str(frame) for frame in frames])

#---------------------------------------------------------------------Output results
if results["status"] == "recognized":
    print(f"Roll Number: {results['roll_number']}")
    print(f"Votes: {results['votes']}/{results['total_predictions']}")
else:
    print("Roll Number: Unknown")
    print(f"Votes: 0/{results['total_predictions']}")
    
