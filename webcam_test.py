import logging
from datetime import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# --- NEW: the added modules (each lives in its own file next to this one) ---
from liveness_detection import LivenessDetector
from person_tracker import Observation, PersonTracker
from risk_calculation import (
    FAKE, HIGH, PENDING, RiskConfig, calculate_risk, display_name,
)
from telegram_alert import TelegramAlerter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)


# ==========================================
# 1. LOAD REGISTERED EMBEDDINGS AND NAMES
# ==========================================

known_embeddings = np.load(
    "known_embeddings.npy"
)

known_names = np.load(
    "known_names.npy"
)

print("Embeddings loaded:", known_embeddings.shape)
print("Registered persons:", np.unique(known_names))


# ==========================================
# 2. LOAD buffalo_l MODEL
# ==========================================

app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "CPUExecutionProvider"
    ]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("Model loaded successfully!")


# ==========================================
# 2b. NEW: LIVENESS, TRACKING, RISK, TELEGRAM
# ==========================================

# Risk rules (odd hours 22:00-06:00, 5 s presence, ...). Change values here, e.g.
# RiskConfig(odd_hour_start=time(23, 0), odd_hour_end=time(5, 0))
risk_config = RiskConfig()

try:
    # reads models/MiniFASNetV2.onnx (created by convert_minifasnet_to_onnx.py)
    liveness_detector = LivenessDetector()
except FileNotFoundError as error:
    print(error)
    exit()

tracker = PersonTracker(
    visit_gap=10.0,            # absent longer than this -> a NEW visit (occurrence + 1)
    grace_gap=2.0,             # short detection gaps do not reset the 5 s timer
    unknown_threshold=0.5,     # same cosine threshold as recognize_face()
    spoof_window=risk_config.spoof_window_seconds
)

alerter = TelegramAlerter(cooldown_seconds=60)   # token/chat id come from .env


# ==========================================
# 3. COSINE SIMILARITY FUNCTION
# ==========================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


# ==========================================
# 4. RECOGNIZE FACE FUNCTION
# ==========================================

def recognize_face(face_embedding):

    similarities = []

    for embedding in known_embeddings:

        similarity = cosine_similarity(
            face_embedding,
            embedding
        )

        similarities.append(similarity)

    # Find best match
    best_index = np.argmax(similarities)

    best_similarity = similarities[
        best_index
    ]

    best_name = known_names[
        best_index
    ]

    # Threshold
    THRESHOLD = 0.5

    if best_similarity >= THRESHOLD:

        return best_name, best_similarity

    else:

        return "UNKNOWN", best_similarity


# ==========================================
# 5. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Error: Cannot open webcam")
    exit()


print("Webcam started!")
print("Press Q to quit.")


# ==========================================
# 6. PROCESS WEBCAM FRAMES
# ==========================================

while True:

    success, frame = cap.read()

    if not success:

        print("Error reading frame")
        break


    now = datetime.now()


    # Detect faces
    faces = app.get(frame)


    # Recognize every face (your existing function, unchanged)
    recognitions = [
        recognize_face(face.embedding)
        for face in faces
    ]


    # NEW: REAL / FAKE for every face (smoothed over the last 5 frames)
    liveness_results = liveness_detector.assess(
        frame,
        [face.bbox for face in faces],
        now
    )


    # NEW: who is who across frames, how long, how many visits
    states = tracker.update(
        [
            Observation(str(name), face.embedding, live.label)
            for face, (name, _), live in zip(
                faces, recognitions, liveness_results
            )
        ],
        now
    )


    # Process every detected face
    for face, (name, similarity), live, state in zip(
        faces, recognitions, liveness_results, states
    ):

        # NEW: risk level for this face
        risk = calculate_risk(
            identity=state.identity,
            liveness=live.label,
            timestamp=now,
            duration=state.duration,
            occurrence_count=state.occurrence_count,
            spoof_attempts=state.spoof_attempts,
            unknown_in_frame=state.unknown_in_frame,
            config=risk_config
        )


        # NEW: Telegram alert (unknown person, or photo of a known person)
        alerter.maybe_alert(
            frame,
            face.bbox,
            state.identity,
            live.label,
            now,
            risk.level,
            risk.triggered_rules
        )


        # Get face bounding box
        bbox = face.bbox.astype(int)

        x1, y1, x2, y2 = bbox


        # Box colour: red = photo, gray = still checking, orange = HIGH risk, green = OK
        if live.label == FAKE:
            color = (0, 0, 255)
        elif live.label == PENDING:
            color = (160, 160, 160)
        elif risk.level == HIGH:
            color = (0, 165, 255)
        else:
            color = (0, 255, 0)


        # Draw rectangle
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        # Display result (name + similarity, exactly as before)
        text = (
            f"{display_name(state.identity)} "
            f"{similarity:.2f}"
        )

        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


        # NEW: liveness (with the model's score for this frame) and risk
        if live.label == PENDING:
            status = "CHECKING..."
        else:
            status = f"{live.label} {live.real_prob:.2f} | {risk.level}"

        cv2.putText(
            frame,
            status,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


    # Show webcam
    cv2.imshow(
        "Known / Unknown Detection",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# 7. RELEASE WEBCAM
# ==========================================

cap.release()

cv2.destroyAllWindows()

alerter.close()   # NEW: let any queued Telegram alert finish sending
