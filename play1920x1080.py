import cv2
import numpy as np
from keras.models import load_model
from random import choice
import os

REV_CLASS_MAP = {
    0: "Rock",
    1: "Paper",
    2: "Scissors",
    3: "None"
}

def mapper(val):
    return REV_CLASS_MAP[val]

def calculate_winner(move1, move2):
    if move1 == move2:
        return "Tie"
    if move1 == "Rock":
        return "User" if move2 == "Scissors" else "Computer"
    if move1 == "Paper":
        return "User" if move2 == "Rock" else "Computer"
    if move1 == "Scissors":
        return "User" if move2 == "Paper" else "Computer"

# Load model
try:
    model = load_model("rock-paper-scissors-model.h5")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("[ERROR] Cannot access webcam.")
    exit()

prev_move = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.rectangle(frame, (140, 140), (520, 520), (0, 255, 0), 3)     # User
    cv2.rectangle(frame, (800, 140), (1180, 520), (0, 255, 0), 3)    # Computer

    roi = frame[140:520, 140:520] 
    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (227, 227))

    pred = model.predict(np.array([img]), verbose=0)
    move_code = np.argmax(pred[0])
    user_move_name = mapper(move_code)
    print(f"[INFO] Predict: {pred[0]} → {user_move_name}")

    if prev_move != user_move_name:
        if user_move_name != "None":
            computer_move_name = choice(["Rock", "Paper", "Scissors"])
            winner = calculate_winner(user_move_name, computer_move_name)
        else:
            computer_move_name = "None"
            winner = "Waiting..."
        prev_move = user_move_name

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Your Move: " + user_move_name, (140, 100), font, 1.5, (0, 255, 255), 2)
    cv2.putText(frame, "Computer: " + computer_move_name, (800, 100), font, 1.5, (0, 255, 255), 2)
    cv2.putText(frame, "Winner: " + winner, (320, 650), font, 2.5, (0, 0, 255), 4)

    icon_path = f"images/{computer_move_name}.png"
    if computer_move_name != "None" and os.path.exists(icon_path):
        icon = cv2.imread(icon_path)
        icon = cv2.resize(icon, (380, 380))
        frame[140:520, 800:1180] = icon

    cv2.imshow("Rock Paper Scissors", frame)

    key = cv2.waitKey(10)
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()


