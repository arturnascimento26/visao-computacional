import cv2
import mediapipe as mp
import numpy as np

# Inicializa utilitários de desenho do MediaPipe globalmente para evitar recriação
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

def process_image(frame, recognizer, clf, label_encoder):
    """
    Recebe uma imagem BGR (OpenCV format), processa para detectar gestos e desenha resultados.
    
    Args:
        frame: Imagem capturada (BGR).
        recognizer: Instância do MediaPipe GestureRecognizer.
        clf: Modelo de classificação customizado (joblib loaded).
        label_encoder: Encoder de labels correspondente ao clf (joblib loaded).
        
    Returns:
        frame: Imagem com anotações e desenhos.
    """
    # 1. Preparação da imagem
    # Flip horizontal para efeito espelho
    frame = cv2.flip(frame, 1)
    # MediaPipe usa RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    # Timestamp necessário para o modo VIDEO do MediaPipe
    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    
    # 2. Extração de Landmarks com MediaPipe
    recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

    if recognition_result.hand_landmarks:
        for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
            # A. Desenha os landmarks na imagem
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # B. Prepara dados para o modelo customizado
            # Extraímos handedness e landmarks [handedness, x0, y0, z0, ..., x20, y20, z20]
            hand_label = recognition_result.handedness[i][0].category_name
            handedness_val = 0 if hand_label == 'Left' else 1
            
            landmarks_array = [handedness_val]
            for lm in hand_landmarks:
                landmarks_array.extend([lm.x, lm.y, lm.z])
            
            # Formato esperado pelo sklearn (2D array: [1, 64])
            features = np.array(landmarks_array).reshape(1, -1)
            
            # C. Predição com o modelo customizado
            prediction_idx = clf.predict(features)[0]
            prediction_prob = np.max(clf.predict_proba(features))
            gesture_name = label_encoder.inverse_transform([prediction_idx])[0]

            # D. Exibindo informações na tela
            color = (0, 255, 0) # Verde
            display_text = f"Custom {hand_label}: {gesture_name} ({prediction_prob:.2f})"
            cv2.putText(frame, display_text, (20, 50 + (i * 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return frame
