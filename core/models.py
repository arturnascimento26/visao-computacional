import os
import joblib
import mediapipe as mp

# Caminhos padrão para os modelos
DEFAULT_MP_MODEL_PATH = "gesture_recognizer.task"
DEFAULT_CUSTOM_MODEL_PATH = "gesture_model.joblib"
DEFAULT_ENCODER_PATH = "label_encoder.joblib"

def load_gesture_models(mp_path=DEFAULT_MP_MODEL_PATH, 
                        custom_path=DEFAULT_CUSTOM_MODEL_PATH, 
                        encoder_path=DEFAULT_ENCODER_PATH):
    """
    Carrega o modelo do MediaPipe, o classificador customizado e o encoder de labels.
    Retorna uma tupla (options, classifier, label_encoder) ou levanta um erro se um arquivo não for encontrado.
    """
    if not all(os.path.exists(p) for p in [mp_path, custom_path, encoder_path]):
        missing = [p for p in [mp_path, custom_path, encoder_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"Arquivos de modelo não encontrados: {missing}")

    # Carrega o modelo customizado e o encoder de labels
    clf = joblib.load(custom_path)
    label_encoder = joblib.load(encoder_path)

    # Configurações do MediaPipe Tasks
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=mp_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return options, clf, label_encoder
