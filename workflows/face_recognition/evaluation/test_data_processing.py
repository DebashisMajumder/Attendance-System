import os
import cv2
from retinaface import RetinaFace, model
import numpy as np
import insightface

#----------------------------------------------------------------------------IMAGE PROCESSING
ip_dir = "/home/debashis/SIH'26/Inter Clg/Attendance-System/data/test_data"
img_op = {}
op_dir = "/home/debashis/SIH'26/Inter Clg/Attendance-System/data/test_data_embeddings"

#----------------------------------------------------------------------------each rollnum folder
for roll in sorted(os.listdir(ip_dir)):

    roll_path = os.path.join(ip_dir, roll)

    # Make sure it is a folder
    if not os.path.isdir(roll_path):
        continue

    print(f"\nProcessing roll number: {roll}")

    # Create list for this roll number
    if roll not in img_op:
        img_op[roll] = []

    for img_name in sorted(os.listdir(roll_path)):

        # Check image extension
        if not img_name.lower().endswith(
            (".png", ".jpg", ".jpeg", ".webp")
        ):
            continue

        img_path = os.path.join(roll_path, img_name)

        print(f"  Processing: {img_name}")
        
        #-------------------------------------------------------------------------------Read the image
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not read image: {img_name}")
            continue
        
        #--------------------------------------------------------------------------------RetinaFace detection
        faces = RetinaFace.detect_faces(img_path)
        
        #-------------------------------------------------------------------------------------No faces detected
        if not faces:
            print(f" No faces detected in {img_name}")
            continue
        
        #------------------------------------------------------------------------------------Process detected faces
        for face_number, (_, face_data) in enumerate(faces.items()):

            x1, y1, x2, y2 = face_data["facial_area"]

            #--------------------------------------------------------------------------------------Add padding
            padding = int(0.2 * (x2 - x1)) #20% padding

            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.shape[1], x2 + padding)
            y2 = min(img.shape[0], y2 + padding)

            #-----------------------------------------------------------------------------------------Crop face
            face_crop = img[y1:y2, x1:x2]
            #-----------------------------------------------------------------------------------------Output filename
            if roll not in img_op:
                 img_op[roll] = []

            img_op[roll].append({
                "frame": img_path,
                "face_number": face_number,
                "face_crop": face_crop
            })
             
            print(f"  Detected face {face_number} in {img_path}")
             
print("\nAll images processed!")

#-----------------------------------------------------------------------------------EMBEDDING
embedding_model_path = "/home/debashis/SIH'26/Inter Clg/Attendance-System/models/w600k_r50.onnx"

gpu_enabled = True #----------------------------------------------------------------gpu enabled check
gpu_ctx_id = 0 if gpu_enabled else -1
print(f"\nEmbedding model loaded! & using {'GPU' if gpu_enabled else 'CPU'}")

os.makedirs(op_dir, exist_ok=True)

#load model
embedding_model = insightface.model_zoo.get_model(embedding_model_path)

# using gpu if available
embedding_model.prepare(ctx_id = gpu_ctx_id)

print(f"Using {'GPU' if gpu_enabled else 'CPU'} for embedding extraction")
print("Model loaded successfully")

# DB
for roll in sorted(img_op.keys()):

    print(f"\nProcessing roll number: {roll}")

    embeddings = []

    # Process every cropped face for this roll

    for face_data in img_op[roll]:

        face_crop = face_data["face_crop"]
        frame = face_data["frame"]
        face_number = face_data["face_number"]

        # Get embedding

        embedding = embedding_model.get_feat(face_crop)

        embedding = embedding.flatten()

        # L2 normalization

        norm = np.linalg.norm(embedding)

        if norm == 0:
            print(
                f"  Invalid embedding: "
                f"{frame}"
            )
            continue

        embedding = embedding / norm

        # Store embedding

        embeddings.append(embedding)

        print(
            f"  {os.path.basename(frame)} | "
            f"Face {face_number} | "
            f"Embedding shape: {embedding.shape}"
        )

    # SAVE EMBEDDINGS

    if len(embeddings) == 0:

        print(
            f"No valid embeddings found "
            f"for roll number: {roll}"
        )

        continue

    embeddings_arr = np.array(
        embeddings,
        dtype=np.float32
    )

    op_path = os.path.join(
        op_dir,
        f"{roll}.npy"
    )

    np.save(
        op_path,
        embeddings_arr
    )

    print(
        f"Saved embeddings for roll number: "
        f"{roll}"
    )

    print(
        f"Path: {op_path}"
    )

    print(
        f"Total embeddings: "
        f"{len(embeddings)}"
    )

    print(
        f"Embedding array shape: "
        f"{embeddings_arr.shape}"
    )


print("Database Embeding Generation Completed!!")