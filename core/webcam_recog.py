import cv2
import mediapipe as mp
import os
from models import load_gesture_models
from processor import process_image

def main():
    try:
        # 1. Carrega os modelos e configurações
        print("--- Carregando modelos ---")
        options, clf, label_encoder = load_gesture_models()
        GestureRecognizer = mp.tasks.vision.GestureRecognizer
    except FileNotFoundError as e:
        print(f"Erro ao carregar modelos: {e}")
        return
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        return

    # 2. Inicializa a captura de vídeo
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível acessar a webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nIniciando reconhecimento CUSTOMIZADO... Pressione 'q' para sair.")

    # 3. Loop principal de processamento
    with GestureRecognizer.create_from_options(options) as recognizer:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignorando frame vazio da webcam.")
                continue

            # Processa o frame usando a função extraída em processor.py
            processed_frame = process_image(frame, recognizer, clf, label_encoder)

            # Exibe o resultado
            cv2.imshow('Custom Gesture Recognition', processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # 4. Finalização
    cap.release()
    cv2.destroyAllWindows()
    print("Finalizado.")

if __name__ == "__main__":
    main()

