import time
from pathlib import Path

from src.face.detector import FaceDetector
from src.face.aligner import FaceAligner
from src.face.embedder import FaceEmbedder

from src.recognition.classifier import FaceClassifier
from src.recognition.identification import FaceIdentifier

from src.pipeline.recognition_pipeline import RecognitionPipeline


class FaceRecognizer:

    def __init__(
        self,
        cam_frame_path,
        embedding_model_path,
        classifier_model_path,
        expected_frames=5,
        gpu_ctx_id=0,
        padding_ratio=0.2,
        detection_confidence=0.5
    ):
        #---------------------------------------------------------------------------------Paths

        self.cam_frame_path = Path(
            cam_frame_path
        )

        self.embedding_model_path = Path(
            embedding_model_path
        )

        self.classifier_model_path = Path(
            classifier_model_path
        )

        self.expected_frames = expected_frames


        #-------------------------------------------------------------------------------Validate paths

        if not self.cam_frame_path.exists():

            raise FileNotFoundError(
                f"Camera frame directory not found: "
                f"{self.cam_frame_path}"
            )


        if not self.embedding_model_path.exists():

            raise FileNotFoundError(
                f"Embedding model not found: "
                f"{self.embedding_model_path}"
            )


        if not self.classifier_model_path.exists():

            raise FileNotFoundError(
                f"Classifier model not found: "
                f"{self.classifier_model_path}"
            )


        #---------------------------------------------------------------------------Initialize face detector

        self.detector = FaceDetector()


        #-----------------------------------------------------------------------------Initialize face aligner

        self.aligner = FaceAligner(
            padding_ratio=padding_ratio
        )


        #------------------------------------------------------------------------------Initialize Embedding model

        self.embedder = FaceEmbedder(
            model_path=str(
                self.embedding_model_path
            ),
            ctx_id=gpu_ctx_id
        )


        #---------------------------------------------------------------------------Initialize KNN classifier

        self.classifier = FaceClassifier(
            model_path=str(
                self.classifier_model_path
            )
        )


        #-----------------------------------------------------------------------------------Create identifier

        self.identifier = FaceIdentifier(
            detector=self.detector,
            aligner=self.aligner,
            embedder=self.embedder,
            classifier=self.classifier
        )


        #-----------------------------------------------------------------------------------Create recognition pipeline

        self.pipeline = RecognitionPipeline(
            identifier=self.identifier,
            min_detection_confidence=detection_confidence
        )


        print(
            "[FaceRecognizer] "
            "Recognition system initialized."
        )


    #-----------------------------------------------------------------------------------GET CAMERA FRAMES

    def _get_frames(self):

        extensions = {
            ".jpg",
            ".jpeg",
            ".png"
        }

        frames = [
            path
            for path in self.cam_frame_path.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in extensions
            )
        ]

        return sorted(frames)


    #-----------------------------------------------------------------------------------------RECOGNIZE

    def recognize(self):

        total_start = time.perf_counter()
        
        frames = self._get_frames()


        print(
            f"[FaceRecognizer] "
            f"Found {len(frames)} frame(s)."
        )


        #-----------------------------------------------------------------------------Check frame count

        if len(frames) < self.expected_frames:

            return {
                "status": "waiting",
                "roll_number": None,
                "votes": 0,
                "total_predictions": 0,
                "frames_received": len(frames),
                "required_frames": self.expected_frames
            }


        #-----------------------------------------------------------------------------------Use latest N frames

        frames = frames[-self.expected_frames:]


        print(
            f"[FaceRecognizer] "
            f"Processing {len(frames)} frames..."
        )

        pipeline_start = time.perf_counter()

        #-------------------------------------------------------------------------Run recognition pipeline

        results = self.pipeline.process(
            [
                str(frame)
                for frame in frames
            ]
        )

        pipeline_time = time.perf_counter() - pipeline_start
        total_time = time.perf_counter() - total_start

        print(f"[TIMING] Pipeline: {pipeline_time:.3f}s")
        print(f"[TIMING] Total recognition: {total_time:.3f}s")
        
        #------------------------------------------------------------------------------Return result

        if results["status"] == "recognized":

            print(
                f"[FaceRecognizer] "
                f"Recognized: "
                f"{results['roll_number']}"
            )

        else:

            print(
                "[FaceRecognizer] "
                "Face not recognized."
            )


        return results