import os
import pickle
import mediapipe as mp
import cv2

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Mediapipe Hands with static image mode
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Define the dataset directory
DATA_DIR = './data'

# Initialize arrays to store the landmark data and corresponding labels
data = []   # Stores the landmark coordinates for each image
labels = [] # Stores the class labels corresponding to the data

# Iterate through each class directory in the dataset
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        
        # Load the image using OpenCV
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)

        # Check if exactly one hand is detected
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
            for i in range(len(results.multi_hand_landmarks[0].landmark)):
                x = results.multi_hand_landmarks[0].landmark[i].x
                y = results.multi_hand_landmarks[0].landmark[i].y
                data_aux.append(x)
                data_aux.append(y)

            # Append data if length matches the expected size (42 in this case)
            if len(data_aux) == 42:
                data.append(data_aux)
                labels.append(dir_)
            else:
                print(f"Skipping image due to unexpected data length: {len(data_aux)}, Path: {os.path.join(dir_, img_path)}")
        else:
            print(f"Skipping image: no hand or multiple hands detected, Path: {os.path.join(dir_, img_path)}")

# Save the filtered data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
