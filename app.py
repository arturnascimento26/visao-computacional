from fasthtml.common import *
import cv2
import numpy as np
import base64
import os
import sys
import json
import mediapipe as mp

# Add current directory to sys.path for core imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models import load_gesture_models
from core.processor import process_image
from core.utils import decode_image, encode_image
import time


# Initialize FastHTML app
app, rt = fast_app(static_path='assets')

# Load models
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MP_PATH = os.path.join(MODELS_DIR, "gesture_recognizer.task")
CUSTOM_PATH = os.path.join(MODELS_DIR, "gesture_model.joblib")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")

try:
    options, clf, label_encoder = load_gesture_models(
        mp_path=MP_PATH,
        custom_path=CUSTOM_PATH,
        encoder_path=ENCODER_PATH
    )
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    recognizer = GestureRecognizer.create_from_options(options)
except Exception as e:
    print(f"Error loading models: {e}")
    # Fallback to current directory for demo if models/ not found
    try:
        options, clf, label_encoder = load_gesture_models()
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
        recognizer = GestureRecognizer.create_from_options(options)
    except:
        sys.exit(1)

@rt("/")
def get():
    return Titled("Hand Gesture Recognition",
        Link(rel="stylesheet", href="/style.css"),
        Div(
            Div(
                # Left Side: Primary View (Large Webcam)
                Div(
                    Div(
                        Video(id="webcam", autoplay=True, width="640", height="480"),
                        Div(
                            Span("FPS: ", cls="fps-label"),
                            Span("0", id="fps-val"),
                            cls="fps-container"
                        ),
                        cls="video-wrapper"
                    ),
                    P("Webcam Feed", cls="feed-label"),
                    cls="primary-view"
                ),
                # Right Side: Sidebar View (Stacked Data + Processed Feed)
                Div(
                    Div(
                        # Controls
                        Div(
                            Div(
                                Label("Quality: ", For="quality-slider"),
                                Input(type="range", id="quality-slider", min="0.1", max="1.0", step="0.1", value="0.5"),
                                Span("0.5", id="quality-val"),
                                cls="control-item"
                            ),
                            Div(
                                Label("Show Landmarks: ", For="landmarks-checkbox"),
                                Input(type="checkbox", id="landmarks-checkbox", checked=True),
                                cls="control-item"
                            ),
                            cls="controls-container"
                        ),
                        Div(
                            Img(id="gesture-img", style="display: none;"),
                            Div(id="labels-container"),
                            cls="result-section"
                        ),
                        P("Live Data", cls="feed-label"),
                        cls="data-container"
                    ),
                    Div(
                        Canvas(id="canvas", width="320", height="240"), # Smaller preview in sidebar
                        P("Processed Feed", cls="feed-label"),
                        cls="feed-container preview"
                    ),
                    cls="sidebar-view"
                ),
                id="workspace",
                cls="workspace-container"
            ),
            id="main-container"
        ),
        Script(src="/script.js")
    )

@app.ws('/ws')
async def ws(msg: str, send):
    last_time = time.time()
    try:
        data = json.loads(msg)
        frame_data = data.get("image")
        show_landmarks = data.get("show_landmarks", True)
        
        # Decode base64 image
        frame = decode_image(frame_data)

        if frame is not None:
            # Process image
            processed_frame, labels, matched_gesture = process_image(frame, recognizer, clf, label_encoder, show_landmarks=show_landmarks)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time
            
            # Encode back to base64
            encoded_img = encode_image(processed_frame)
            
            # Send as JSON
            await send(json.dumps({
                "image": encoded_img,
                "labels": labels,
                "matched_gesture": matched_gesture,
                "fps": round(float(fps), 1)
            }))
    except Exception as e:
        # Fallback for old clients sending raw string if any
        try:
            frame = decode_image(msg)
            if frame is not None:
                processed_frame, labels, matched_gesture = process_image(frame, recognizer, clf, label_encoder)
                encoded_img = encode_image(processed_frame)
                await send(json.dumps({"image": encoded_img, "labels": labels, "matched_gesture": matched_gesture}))
        except:
            print(f"WS Error: {e}")

serve()
