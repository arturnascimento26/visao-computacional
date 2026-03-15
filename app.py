from fasthtml.common import *
import cv2
import numpy as np
import base64
import os
import sys
import mediapipe as mp

# Add current directory to sys.path for core imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.models import load_gesture_models
from core.processor import process_image

# Initialize FastHTML app
app, rt = fast_app()

# Load models
try:
    options, clf, label_encoder = load_gesture_models()
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    recognizer = GestureRecognizer.create_from_options(options)
except Exception as e:
    print(f"Error loading models: {e}")
    # In a real app, you might want to handle this more gracefully
    sys.exit(1)

@rt("/")
def get():
    return Titled("Hand Gesture Recognition",
        Video(id="webcam", autoplay=True, width="640", height="480", style="display:none"),
        Canvas(id="canvas", width="640", height="480"),
        Script("""
            const video = document.getElementById('webcam');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const ws = new WebSocket(`ws://${window.location.host}/ws`);

            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => { video.srcObject = stream; })
                .catch(err => console.error("Error accessing webcam:", err));

            ws.onmessage = (event) => {
                const img = new Image();
                img.onload = () => ctx.drawImage(img, 0, 0);
                img.src = 'data:image/jpeg;base64,' + event.data;
            };

            function sendFrame() {
                if (ws.readyState === WebSocket.OPEN) {
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    const data = canvas.toDataURL('image/jpeg', 0.5).split(',')[1];
                    ws.send(data);
                }
                requestAnimationFrame(sendFrame);
            }
            video.onplay = () => requestAnimationFrame(sendFrame);
        """)
    )

@app.ws('/ws')
async def ws(msg: str, send):
    try:
        # Decode base64 image
        img_data = base64.b64decode(msg)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is not None:
            # Process image
            processed_frame = process_image(frame, recognizer, clf, label_encoder)
            
            # Encode back to base64
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
            encoded_img = base64.b64encode(buffer).decode('utf-8')
            
            await send(encoded_img)
    except Exception as e:
        print(f"WS Error: {e}")

serve()
