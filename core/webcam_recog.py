import cv2
import mediapipe as mp
import numpy as np
import os
import joblib

# Caminhos para os modelos
MP_MODEL_PATH = "gesture_recognizer.task" # Usado apenas para extrair landmarks
CUSTOM_MODEL_PATH = "gesture_model.joblib"
ENCODER_PATH = "label_encoder.joblib"

def main():
    # Verifica se os modelos existem
    if not all(os.path.exists(p) for p in [MP_MODEL_PATH, CUSTOM_MODEL_PATH, ENCODER_PATH]):
        print("Erro: Um ou mais arquivos de modelo não foram encontrados (.task, .joblib).")
        return

    # Carrega o modelo customizado e o encoder de labels
    print("--- Carregando modelos customizados ---")
    clf = joblib.load(CUSTOM_MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    # Inicializa o modelo do MediaPipe Tasks
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    # Configurações do MediaPipe (usaremos apenas para landmarks)
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    mp_hands = mp.tasks.vision.HandLandmarksConnections
    mp_drawing = mp.tasks.vision.drawing_utils
    mp_drawing_styles = mp.tasks.vision.drawing_styles

    print("\nIniciando reconhecimento CUSTOMIZADO... Pressione 'q' para sair.")

    with GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            
            # Extrai landmarks usando MediaPipe
            recognition_result = recognizer.recognize_for_video(mp_image, timestamp_ms)

            if recognition_result.hand_landmarks:
                for i, hand_landmarks in enumerate(recognition_result.hand_landmarks):
                    # 1. Desenha os landmarks
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                    # 2. Prepara dados para o modelo customizado
                    # Criamos um vetor flat [handedness, x0, y0, z0, ..., x20, y20, z20]
                    hand_label = recognition_result.handedness[i][0].category_name
                    handedness_val = 0 if hand_label == 'Left' else 1
                    
                    landmarks_array = [handedness_val]
                    for lm in hand_landmarks:
                        landmarks_array.extend([lm.x, lm.y, lm.z])
                    
                    # Converte para o formato esperado pelo sklearn (2D array: [1, 64])
                    features = np.array(landmarks_array).reshape(1, -1)
                    
                    # Predição do modelo customizado
                    prediction_idx = clf.predict(features)[0]
                    prediction_prob = np.max(clf.predict_proba(features))
                    gesture_name = label_encoder.inverse_transform([prediction_idx])[0]

                    # 3. Exibe o resultado
                    hand_label = recognition_result.handedness[i][0].category_name
                    color = (0, 255, 0) # Verde para custom model
                    
                    display_text = f"Custom {hand_label}: {gesture_name} ({prediction_prob:.2f})"
                    cv2.putText(frame, display_text, (20, 50 + (i * 40)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow('Custom Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

