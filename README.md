# AI-Powered Smart Attendance System

An automated classroom attendance platform that combines **fingerprint identification** with **face recognition** to eliminate proxy attendance and reduce manual attendance overhead.

Fingerprint identification establishes a student's claimed identity and triggers image capture. The face-recognition pipeline independently determines the identity from the captured frames. Attendance is recorded only when both identities agree.

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Core Design Principle](#core-design-principle)
- [High-Level Architecture](#high-level-architecture)
- [Hardware Architecture](#hardware-architecture)
- [Software Architecture](#software-architecture)
- [Technology Stack](#technology-stack)
- [Face Recognition Pipeline](#face-recognition-pipeline)
- [Five-Frame Recognition Strategy](#five-frame-recognition-strategy)
- [Fingerprint-to-Face Verification](#fingerprint-to-face-verification)
- [Attendance Session Model](#attendance-session-model)
- [Complete Data Flow](#complete-data-flow)
- [Flask Server & API Routes](#flask-server--api-routes)
- [Database Architecture](#database-architecture)
- [Frame Lifecycle](#frame-lifecycle)
- [Security and Privacy Model](#security-and-privacy-model)
- [Attendance Decision Logic](#attendance-decision-logic)
- [Evaluation Methodology](#evaluation-methodology)
- [Error Handling](#error-handling)
- [Web Application Workflow](#web-application-workflow)
- [Reporting](#reporting)
- [Advantages](#advantages)
- [Limitations of the Current Prototype](#limitations-of-the-current-prototype)
- [Future Improvements](#future-improvements)
- [Summary](#summary)

---

## Executive Summary

The system uses two independent biometric signals to verify student identity:

- **Fingerprint identification** — matched locally on the fingerprint sensor against enrolled templates
- **Face recognition** — computed server-side from live camera frames

Once a fingerprint is matched, its associated enrollment ID is sent to the server and treated as the student's roll number. This event triggers the camera subsystem to capture five frames of the student's face. The frames are uploaded to the Flask server, where the face-recognition pipeline generates a predicted roll number using an InsightFace Buffalo_L embedding model and a KNN classifier.

The server then compares the two identities:

```
Fingerprint Roll Number
        |
Face Recognition Roll Number
        |
      Compare
     /       \
   Match    Mismatch
    |          |
Present      Reject
```

Attendance is recorded only when:

```
Fingerprint ID == Face Recognition ID
```

The attendance date, subject, and class are selected by the teacher through the web interface. The biometric hardware is not responsible for attendance context — its role is limited to identification and frame capture.

---

## Problem Statement

Traditional classroom attendance systems have several limitations:

- Manual attendance consumes classroom time
- Proxy attendance is possible
- Paper records are inefficient to maintain
- Attendance records are difficult to analyze
- Single-signal biometric systems provide only one source of identification
- Hardware-generated timestamps can conflict with the date selected by the instructor

This system addresses these issues by combining biometric identification, computer vision, machine learning, embedded hardware, and centralized attendance management.

---

## Project Objectives

- Automate classroom attendance
- Reduce proxy attendance
- Combine fingerprint and face recognition for identity verification
- Keep fingerprint biometric data stored locally on the sensor
- Use fingerprint identification to trigger image acquisition
- Process multiple face frames instead of relying on a single image
- Digitally store attendance records
- Allow teachers to select the attendance date and subject
- Provide department, subject, student, and attendance management
- Generate attendance reports
- Provide CSV and Excel export functionality
- Integrate embedded hardware with an AI-based server
- Provide a scalable architecture for future deployment

---

## Core Design Principle

The most important architectural decision is the **separation between identity selection and identity verification**.

**Fingerprint subsystem** — answers *"Which enrolled student placed their finger?"* and returns an enrollment ID. In the current prototype:

```
Fingerprint Enrollment ID = Student Roll Number
```

**Face-recognition subsystem** — answers *"Whose face appears in the captured frames?"* and independently predicts a roll number.

**Final decision** — the server answers *"Do both biometric systems identify the same student?"* Only then is attendance marked.

```
Fingerprint Identity
        +
Face Identity
        |
Identity Agreement
        |
Attendance
```

This creates a two-factor biometric verification mechanism.

---

## High-Level Architecture

```
                    +--------------------------+
                    |         Teacher          |
                    |  Attendance Web Interface |
                    +-------------+------------+
                                  |
                          Select Date
                          Select Subject
                          Select Class
                                  |
                                  v
                    +--------------------------+
                    |       Flask Server       |
                    |     Application Layer    |
                    +-------------+------------+
                                  |
                +-----------------+-----------------+
                |                                    |
                v                                    v
       +------------------+                +------------------+
       | Fingerprint      |                | ESP32-CAM        |
       | Sensor           |                | (Camera + Wi-Fi) |
       +--------+---------+                +---------+--------+
                |                                     |
          Enrollment ID                          Five Frames
                |                                     |
                +------------------+------------------+
                                   v
                        +---------------------+
                        | Frame Storage       |
                        | cam_frames/         |
                        +----------+----------+
                                   |
                                   v
                        +---------------------+
                        | Face Recognition    |
                        | Pipeline            |
                        +----------+----------+
                                   |
                                   v
                        +---------------------+
                        | InsightFace         |
                        | Buffalo_L           |
                        | Embedding Model     |
                        +----------+----------+
                                   |
                                   v
                        +---------------------+
                        | KNN Classifier      |
                        +----------+----------+
                                   |
                            Face Roll Number
                                   |
                +------------------+-------------------+
                |      Identity Comparison              |
                |   Fingerprint ID == Face ID ?          |
                +------------------+---------------------+
                                   |
                     +-------------+--------------+
                     |                            |
                   MATCH                      MISMATCH
                     |                            |
                     v                            v
               Mark Present                    Reject
                     |
                     v
              SQLite Database
                     |
                     v
              Attendance Report
```

---

## Hardware Architecture

The hardware layer is responsible for biometric acquisition, image acquisition, and communication with the server.

### Fingerprint Sensor

Fingerprint templates are stored locally on the sensor. The server never receives or stores the fingerprint image or template itself.

```
Student places finger
        |
Fingerprint sensor scans finger
        |
Sensor compares against local templates
        |
Match found
        |
Enrollment ID returned
```

For the current implementation, `Enrollment ID = Roll Number`. For example, fingerprint ID `23` corresponds to student roll number `23`.

This design eliminates the need for the server to maintain fingerprint biometric data.

### Camera Subsystem

The camera captures the student's face after successful fingerprint identification, acquiring **5 frames per attendance attempt**.

The camera does not run the face-recognition pipeline itself. Its responsibility is limited to **Capture → Upload**, keeping the embedded device lightweight while the more computationally intensive machine-learning processing remains on the server.

### ESP32-CAM Controller

The ESP32-CAM module serves as both the camera and the hardware-side controller, responsible for:

- Managing communication between the fingerprint sensor and the server
- Receiving fingerprint match events
- Controlling the camera trigger
- Capturing the five-frame burst
- Connecting to the server over Wi-Fi and uploading frames
- Sending the fingerprint enrollment ID

Using a single ESP32-CAM module for both image capture and device-side control keeps the hardware footprint small and avoids the need for a separate embedded controller board.

### Power Supply

The device is powered by a **lithium-ion (Li-ion) battery**, allowing the fingerprint-and-camera unit to operate as a standalone, portable station independent of a fixed power outlet. This is particularly relevant for classroom deployment, where the unit may need to be mounted or moved without immediate access to mains power.

---

## Software Architecture

```
+----------------------------------------+
| Presentation Layer                     |
| HTML / CSS / JavaScript                |
+--------------------+--------------------+
                     |
+--------------------v--------------------+
| Application Layer                       |
| Flask                                   |
| Authentication / Routes / Logic         |
+--------------------+--------------------+
                     |
+--------------------v--------------------+
| Recognition Layer                       |
| InsightFace + Buffalo_L + KNN           |
+--------------------+--------------------+
                     |
+--------------------v--------------------+
| Data Layer                              |
| SQLAlchemy + SQLite                     |
+------------------------------------------+
```

---

## Technology Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python, Flask, Flask-SQLAlchemy, SQLite, Werkzeug |
| **Machine Learning** | InsightFace, Buffalo_L, ONNX Runtime, KNN classifier, NumPy, OpenCV |
| **Preprocessing** | RetinaFace (face detection and cropping) |
| **Frontend** | HTML, CSS, JavaScript, Jinja2 templates |
| **Hardware** | ESP32-CAM (camera + Wi-Fi controller), fingerprint sensor, Li-ion battery (portable power supply), optional TFT/status display |
| **Reporting** | OpenPyXL, CSV generation |

---

## Face Recognition Pipeline

The face-recognition system is divided into two phases: training/preprocessing, and runtime recognition.

### Training / Preprocessing

```
Training Images
      |
RetinaFace
      |
Face Detection
      |
Face Crop
      |
Buffalo_L
      |
Face Embedding
      |
KNN Training
      |
KNN Model
```

### Runtime Recognition

```
Five Camera Frames
        |
Frame Preparation
        |
Face Alignment / Padding
        |
Buffalo_L
        |
Face Embedding
        |
KNN Prediction
        |
Five Predictions
        |
Majority Voting
        |
Face Roll Number
```

### Why RetinaFace Is Used During Preprocessing

RetinaFace is used primarily during database preparation, not as the real-time attendance detector. The training dataset is organized by student:

```
train_data/
|
+-- 01/
|   +-- image1.jpg
|   +-- image2.jpg
|   +-- ...
|
+-- 02/
|   +-- image1.jpg
|   +-- image2.jpg
|   +-- ...
|
+-- 03/
    +-- image1.jpg
    +-- image2.jpg
    +-- ...
```

RetinaFace detects the face in each image; the detected face is cropped and stored for embedding generation:

```
Original Training Image -> RetinaFace -> Detected Face -> Cropped Face -> Embedding Generation
```

### InsightFace Buffalo_L

The recognition system uses the Buffalo_L model, specifically `w600k_r50.onnx`, to convert a face image into a numerical embedding:

```
Face Image -> Buffalo_L -> [0.12, -0.43, 0.81, ...]
```

The embedding represents facial characteristics in a high-dimensional numerical space. The classifier operates on these embeddings, not on the raw image.

### KNN Classifier

Generated embeddings are used to train a K-Nearest Neighbors classifier.

**Training:**
```
Student Face -> Face Embedding -> Embedding Vector + Roll Number -> KNN Training Dataset -> KNN Model
```

**Recognition:**
```
Unknown Face -> Embedding -> KNN -> Nearest Training Samples -> Predicted Roll Number
```

The KNN classifier performs identity classification after the neural network generates the facial representation.

---

## Five-Frame Recognition Strategy

Instead of trusting a single frame, the system captures five frames and applies majority voting:

```
Frame 1 -> Roll 23
Frame 2 -> Roll 23
Frame 3 -> Roll 23
Frame 4 -> Roll 18
Frame 5 -> Roll 23

Roll 23 = 4 votes
Roll 18 = 1 vote

Final prediction = Roll 23
```

The configured minimum agreement threshold is:

```
RECOGNITION_MIN_AGREEMENT = 3
```

At least three agreeing predictions are required before a face identity is accepted, reducing the effect of a single incorrectly classified frame.

---

## Fingerprint-to-Face Verification

This is the central concept of the system.

**Match example:**
```
Fingerprint sensor -> ID 23
Face recognition -> Roll 23

23 == 23  ->  Identity verified -> Attendance marked
```

**Mismatch example:**
```
Fingerprint sensor -> ID 23

Face recognition votes:
23, 23, 18, 18, 18

Majority prediction = 18

23 != 18  ->  Attendance rejected
```

This prevents a student from authenticating with another student's fingerprint while presenting a different person's face.

---

## Attendance Session Model

The attendance date belongs to the **attendance session**, not the biometric device. The teacher selects the date, subject, department, and year through the web interface before students authenticate:

```
Attendance Session
|
+-- Date
+-- Subject
+-- Department
+-- Year
```

Example:
```
Session:
Date = 25 Aug 2026
Subject = Machine Learning
Department = CSE
Year = 3rd Year

Student authenticates:
Fingerprint ID = 23
Face ID = 23

Server creates:
Student = Roll 23
Subject = Machine Learning
Date = 25 Aug 2026
Status = Present
```

The fingerprint device only provides the fingerprint ID; the camera subsystem only provides five frames. The server already has the attendance date, subject, and class from the active session — the embedded device is never responsible for maintaining classroom attendance dates.

---

## Complete Data Flow

```
Teacher
  |
  +-- Select Date
  +-- Select Subject
  +-- Select Department
  +-- Select Year
          |
          v
     Flask Server
          |
Student places finger
          |
          v
 Fingerprint Sensor
          |
          v
 Local fingerprint matching
          |
          v
 Enrollment ID / Roll Number
          |
          v
 Camera Trigger
          |
          v
 Capture 5 Frames
          |
          v
 HTTP Upload
          |
          v
 Flask /upload
          |
          v
 cam_frames/
          |
          v
 FaceRecognizer
          |
          v
 Buffalo_L
          |
          v
 Face Embeddings
          |
          v
 KNN
          |
          v
 Five Predictions
          |
          v
 Majority Voting
          |
          v
 Face Roll Number
          |
          v
 Compare: Fingerprint Roll == Face Roll
          |
       +--+--+
       |     |
      YES    NO
       |     |
       v     v
   Present  Reject
       |
       v
 SQLite Attendance
```

---

## Flask Server & API Routes

The Flask application is the central control layer, providing:

**Authentication**
- `POST /login`
- `GET /logout`
- `POST /register`

**Dashboard**
- `GET /`

**Department management**
- `GET /departments`
- `POST /departments/add`
- `GET /departments/delete/<id>`

**Subject management**
- `GET /subjects`
- `POST /subjects/add`
- `GET /subjects/delete/<id>`

**Student management**
- `GET /students`
- `POST /students/add`
- `GET|POST /students/edit/<id>`
- `GET /students/delete/<id>`

**Attendance**
- `GET|POST /attendance`

**Hardware communication**
- `POST /upload`
- `POST /fingerprint`

**Reports**
- `GET /reports`
- `GET /reports/export`
- `GET /reports/export.csv`
- `GET /reports/student/<id>`

### `/upload`

Receives a single camera frame per request. The uploaded image is validated and saved into `cam_frames/`. Once the expected number of frames has been received, recognition is triggered by the `/fingerprint` event rather than by the upload itself, ensuring both biometric signals are available before a decision is made.

### `/fingerprint`

Represents the fingerprint-triggered recognition event. Request body:

```json
{
    "roll_no": "23",
    "subject_id": 1
}
```

The server then:

1. Receives the fingerprint identity
2. Runs face recognition on the frames in `cam_frames/`
3. Checks whether the face-recognition pipeline produced a confident match
4. Retrieves the predicted face roll number
5. Compares the fingerprint and face identities
6. Looks up the student record
7. Checks for an existing attendance record for the session
8. Creates the attendance record if one does not already exist
9. Clears the captured frames

---

## Database Architecture

The database uses SQLite through SQLAlchemy.

```
Department
   |
   +---------- Students
   |
   +---------- Subjects

Student
   |
   +---------- Attendance

Subject
   |
   +---------- Attendance
```

### User

| Field | Description |
|---|---|
| `id` | Primary key |
| `username` | Unique login username |
| `password_hash` | Hashed password (Werkzeug) |
| `role` | `admin` or `teacher` |

### Department

| Field | Description |
|---|---|
| `id` | Primary key |
| `name` | Department name |

A department can have multiple students and multiple subjects.

### Subject

| Field | Description |
|---|---|
| `id` | Primary key |
| `name` | Subject name |
| `code` | Unique subject code |
| `department_id` | Foreign key to Department |
| `semester` | Semester label |

### Student

| Field | Description |
|---|---|
| `id` | Primary key |
| `roll_no` | Unique roll number (also used as the fingerprint enrollment ID) |
| `name` | Student name |
| `email` | Student email |
| `phone` | Student phone number |
| `department_id` | Foreign key to Department |
| `year` | Academic year |

### Attendance

| Field | Description |
|---|---|
| `id` | Primary key |
| `student_id` | Foreign key to Student |
| `subject_id` | Foreign key to Subject |
| `date` | Attendance date |
| `status` | `Present`, `Absent`, or `Late` |

A unique constraint on `(student_id, subject_id, date)` prevents duplicate attendance records — a student can have only one attendance entry per subject per day. If a student authenticates twice, the second attempt detects the existing record rather than creating a duplicate.

---

## Frame Lifecycle

The frame directory is temporary storage, cleared after every recognition event:

```
Capture -> Upload -> cam_frames/ -> Recognition -> Attendance decision -> Delete frames
```

This prevents old frames from being accidentally reused in a later attendance event.

---

## Security and Privacy Model

Fingerprint biometric templates are never transferred to the server:

```
Fingerprint Sensor -- Enrollment ID only --> Server
```

rather than:

```
Fingerprint Sensor -- Raw fingerprint/template --> Server
```

The server stores the student's identity and attendance records, not the fingerprint itself, reducing the amount of biometric information held centrally.

---

## Attendance Decision Logic

```
START
  |
  v
Fingerprint detected?
  |
  +-- NO -> Wait
  |
  +-- YES
        |
        v
Receive fingerprint ID
        |
        v
Capture 5 frames
        |
        v
Enough frames?
        |
        +-- NO -> Wait
        |
        +-- YES
              |
              v
        Run face recognition
              |
              v
        Face recognized?
              |
        +-----+-----+
       NO           YES
        |             |
      Reject          v
                 Get face ID
                      |
                      v
             Fingerprint ID == Face ID?
                      |
                +-----+-----+
               NO           YES
                |             |
              Reject          v
                       Student exists?
                            |
                       +----+----+
                      NO         YES
                       |           |
                     Reject        v
                         Check existing attendance
                                   |
                                   v
                           Mark Present
```

---

## Evaluation Methodology

### False Acceptance and False Rejection

Because the system uses two biometric signals, there are two error categories to evaluate:

**False Acceptance** — a person is incorrectly accepted as another student (e.g., the fingerprint and face IDs coincidentally match due to a face-model misclassification). The KNN confidence threshold and multi-frame agreement requirement are intended to reduce this risk.

**False Rejection** — a legitimate student is rejected because the fingerprint and face predictions disagree. This can occur due to poor lighting, face angle, motion, occlusion, poor camera quality, insufficient training images, similar facial features, or preprocessing errors.

### Confusion Matrix

The face-recognition classifier should be evaluated using a confusion matrix generated from a **separate test dataset**, not the training images:

```
                 Predicted
              01   02   03   04
Actual 01     TP   .    .    .
       02     .    TP   .    .
       03     .    .    TP   .
       04     .    .    .    TP
```

The diagonal represents correct classifications; off-diagonal entries represent misclassification (e.g., actual roll 23 predicted as roll 18).

### Evaluation Metrics

| Metric | Formula |
|---|---|
| Accuracy | Correct Predictions / Total Predictions |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 Score | 2 × Precision × Recall / (Precision + Recall) |

Accuracy alone should not be the only evaluation metric, since the system must distinguish between many individual students. A confusion matrix provides additional insight into which identities are being confused.

### Recommended Evaluation Procedure

The dataset should be split into training and test sets that do not overlap, for example:

```
20 images/student
Training: 15 images
Testing:  5 images
```

```
Test Image -> Buffalo_L -> Embedding -> KNN -> Predicted Student
```

The predicted student is compared against the actual student to generate the confusion matrix and evaluation metrics.

### Why Multiple Frames Improve Reliability

A single frame can be misclassified. Using five independent observations is more robust than relying on one frame:

```
Frame 1 -> 23
Frame 2 -> 23
Frame 3 -> 23
Frame 4 -> 23
Frame 5 -> 19

23 -> 4 votes
19 -> 1 vote

Accepted: 23
```

---

## Error Handling

| Condition | Response |
|---|---|
| Missing fingerprint ID | `fingerprint_id required` |
| Missing image | `file required` |
| Unsupported image type | `unsupported file type` |
| Insufficient frames | `waiting` |
| Face not recognized | `face_not_recognized` |
| Fingerprint-face mismatch | `mismatch` |
| Student not registered | `roll_number_not_enrolled` |

These machine-readable responses allow the hardware or frontend to determine the outcome of an attendance attempt.

### API Communication

```
Hardware -- HTTP POST --> Flask API
```

Fingerprint event body:
```json
{
    "roll_no": "23",
    "subject_id": 1
}
```

Success response:
```json
{
    "ok": true,
    "result": "present",
    "roll_number": "23"
}
```

Rejection response:
```json
{
    "ok": true,
    "result": "invalid",
    "reason": "mismatch"
}
```

---

## Web Application Workflow

1. **Login** — teacher authenticates through the login system
2. **Select attendance** — teacher opens the attendance page
3. **Configure session** — teacher selects department, year, subject, and date
4. **Start attendance** — the system now has the attendance context
5. **Student authentication** — student places a finger on the sensor
6. **Face capture** — the camera captures five frames
7. **Recognition** — the server runs the ML pipeline
8. **Identity comparison** — fingerprint and face identities are compared
9. **Database update** — attendance is marked as Present on a match
10. **UI/report update** — the teacher can view attendance and generate reports

---

## Reporting

Reports can be filtered by subject, year, date range, and student. The system calculates:

- Total Days
- Present
- Absent
- Late
- Attendance Percentage

Reports can be exported to Excel and CSV.

### Attendance Percentage

Both **Present** and **Late** count toward the attendance percentage:

```
Attendance % = (Present + Late) / Total Records × 100
```

Example:
```
Total = 20
Present = 16
Late = 2
Absent = 2

Attendance = (16 + 2) / 20 × 100 = 90%
```

---

## Advantages

- **Dual biometric verification** — the system does not depend entirely on one biometric method
- **Local fingerprint storage** — fingerprint templates remain inside the fingerprint sensor
- **Server-side AI processing** — computationally intensive face recognition is handled by the server, not the embedded camera
- **Multi-frame recognition** — five frames provide multiple observations of the same student
- **Majority voting** — the system tolerates an incorrect individual frame
- **Date-controlled attendance** — the teacher controls the attendance date from the web application
- **Duplicate protection** — database constraints prevent repeated attendance entries
- **Digital reporting** — attendance records can be exported and analyzed
- **Modular architecture** — the fingerprint, camera, recognition, database, and web layers are separated

---

## Limitations of the Current Prototype

**Fingerprint ID and roll number are currently assumed to be identical.** A production deployment could maintain a separate mapping between fingerprint enrollment IDs and student records.

**Face recognition depends on dataset quality**, including the number of training images, lighting, face pose, image quality, dataset diversity, and camera positioning. The current architecture relies on the student positioning their face appropriately within the camera frame rather than running a continuous, computationally expensive detector.

**Hardware communication reliability** — network failures can interrupt frame uploads or fingerprint communication.

**ML model evaluation** — final system accuracy should be established experimentally using an independent test dataset and reported using a confusion matrix, accuracy, precision, recall, and F1 score.

---

## Future Improvements

- Dedicated fingerprint-ID-to-roll-number mapping
- Improved liveness detection
- Better face-quality assessment
- Automated session management
- Real-time attendance dashboard
- Hardware status display
- Network failure recovery
- Better camera positioning and illumination
- Centralized deployment instead of a single local server
- More extensive ML evaluation
- Automated model retraining when new student data is enrolled
- Audit logs for attendance events
- Secure HTTPS communication between embedded devices and server

These are proposed extensions and are not required for the core architecture.

---

## Summary

```
             FINGERPRINT
                  |
                  v
          Student Identity
          (Enrollment ID)
                  |
                  v
              5 FRAMES
                  |
                  v
           FACE RECOGNITION
                  |
          +-------+--------+
          |                |
      Buffalo_L             |
          |                |
      Embedding             |
          |                |
          v                |
         KNN                |
          |                |
          v                |
      Face Identity        |
          |                |
          +-------+--------+
                  v
             COMPARISON
                  |
          +-------+--------+
          |                |
        MATCH           MISMATCH
          |                |
          v                v
      PRESENT            REJECT
          |
          v
    ATTENDANCE DATABASE
          |
          v
       REPORTING
```

The fundamental principle of the project: **fingerprint establishes the claimed identity, face recognition independently verifies that identity, and the server records attendance only when both biometric identities agree within the selected attendance session.**

This gives the project a clear separation of responsibilities — the fingerprint sensor performs local biometric identification, the camera performs image acquisition, the server performs AI-based face verification, and the database performs attendance management and record keeping.
