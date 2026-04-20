from deepface import DeepFace

MODEL_NAME = "ArcFace"
DETECTOR_BACKEND = "opencv"


def get_embedding(image_input, enforce_detection=True):
    """
    Accepts an image path or a numpy frame (BGR) and returns an ArcFace embedding.
    """
    try:
        embedding = DeepFace.represent(
            img_path=image_input,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=enforce_detection,
            align=True,
        )
        return embedding[0]["embedding"]
    except Exception as e:
        print("Error:", e)
        return None


if __name__ == "__main__":
    emb = get_embedding("images/test.jpg")
    
    if emb:
        print("Embedding length:", len(emb))
    else:
        print("No face detected")
