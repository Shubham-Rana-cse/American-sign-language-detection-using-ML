import warnings
import cv2
import mediapipe as mp
import pickle
import numpy as np

# Suppress warnings from google.protobuf
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

# Load the pre-trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Load the reference ASL chart
asl_chart = cv2.imread('./ASL.png')  # Update the path to your ASL image
asl_chart = cv2.resize(asl_chart, (600, 600))  # Resize to make it manageable for display

# Initialize default webcam
cap = cv2.VideoCapture(0)

# Mediapipe initialization
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize Mediapipe Hands with static mode
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Gesture label mapping
labels_dict = {
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'H': 'H', 'I': 'I',
    'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R',
    'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'
}

while True:
    data_aux = []  # Stores landmark data for the current frame
    x_ = []  # Stores x-coordinates for bounding box
    y_ = []  # Stores y-coordinates for bounding box

    ret, frame = cap.read()  # Capture frame from webcam

    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    if not ret:
        break  # Exit loop if frame not captured

    H, W, _ = frame.shape  # Frame dimensions

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for Mediapipe

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:  # If hands are detected
        # Process only the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand landmarks on the frame
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Extract and store normalized x, y coordinates for the first hand
        for i in range(len(hand_landmarks.landmark)):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            data_aux.append(x)
            data_aux.append(y)
            x_.append(x)
            y_.append(y)

        # Calculate bounding box in pixel coordinates
        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        # Predict gesture using the model
        if len(data_aux) == 42:  # Ensure there are exactly 42 features
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[prediction[0]]  # int(prediction[0]) when predicting digits
            accuracy = np.max(model.predict_proba([np.asarray(data_aux)])) * 100  # Get prediction accuracy

            print(f"{predicted_character}: {accuracy:.2f}%")  # Print the predicted gesture with accuracy

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Text to display
            text = f"{predicted_character}: {accuracy:.2f}%"

            # Get text size for the background
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

            # Draw a filled rectangle for the text background above the bounding box
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)

            # Add the text on top of the filled rectangle
            cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 2, cv2.LINE_AA)
        else:
            print("Skipping prediction: Invalid number of landmarks.")

    # Combine the ASL chart and the webcam feed side by side
    combined_frame = np.hstack((cv2.resize(asl_chart, (700, 600)), cv2.resize(frame, (750, 600))))

    # Display the combined frame
    cv2.imshow('ASL Reference & Webcam Feed', combined_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit if 'esc' is pressed
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
