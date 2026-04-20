import cv2
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.liveness import detect_blink

cap = cv2.VideoCapture(0)

print("Blink to verify liveness...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if detect_blink(frame):
        print("Liveness confirmed!")
        break

    cv2.imshow("Liveness Check", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
