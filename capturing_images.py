import os
import cv2

# Define the directory where data will be stored
DATA_DIR = './data'

# Create the data directory if it doesn't exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes for the dataset and the size of each class dataset
number_of_classes = 26
dataset_size = 1500
# Initialize the default webcam for capturing video frames
cap = cv2.VideoCapture(0)

interrupt = -1

# Loop through each class to collect data
for j in range(number_of_classes):
    # Create a directory for the current class if it doesn't exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Wait for user readiness before starting data collection
    while True:
        ret, frame = cap.read()  # Capture a single frame from the webcam

        # Simulating mirror image
        frame = cv2.flip(frame, 1)
        
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Display instructions on the frame
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Show the frame in a window

        # Wait for the user to press 'Q' to proceed or 'ESC' to quit
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        elif key == 27:  # Escape key
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # Counter to keep track of the number of images collected
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a frame from the webcam
        
        # Simulating mirror image
        frame = cv2.flip(frame, 1)
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Show the frame in a window
        cv2.imshow('frame', frame)
        cv2.waitKey(25)  # Wait for a short delay

        # Save the frame as an image in the appropriate class directory
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)

        counter += 1  # Increment the counter


# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
