import cv2
import numpy as np
import base64

def decode_image(msg):
    """
    Decodes base64 string to OpenCV BGR image.
    """
    img_data = base64.b64decode(msg)
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return frame

def encode_image(frame, quality=50):
    """
    Encodes OpenCV BGR image to base64 string.
    """
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    return encoded_img
