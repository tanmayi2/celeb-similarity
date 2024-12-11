import cv2
import os
import numpy as np

# Placeholder function for processing the image
def process_image(image):
    # Create dummy processed images for demonstration
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image, 100, 200)
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    return gray, edges, blurred

# Create a directory to store the frames
output_dir = "data/captured_frames"
os.makedirs(output_dir, exist_ok=True)

# Video capture from the default camera
capture = cv2.VideoCapture(0)

if not capture.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    ret, frame = capture.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break

    # Show the live camera feed
    cv2.imshow('Camera Feed', frame)

    # Key handling
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Save and process the frame when 's' is pressed
        # Save the captured frame

        frame_filename = os.path.join(output_dir, "captured_image.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Frame saved as {frame_filename}")

        # Process the image and generate outputs
        celeb1, celeb2, celeb3 = process_image(frame)  # change the gray 

        # Display all four images
        combined_display = np.hstack((
            frame,  # Original
            cv2.cvtColor(celeb1, cv2.COLOR_GRAY2BGR),  # image of output celeb1 here
            cv2.cvtColor(celeb2, cv2.COLOR_GRAY2BGR),  # image of output celeb2 here
            celeb3  # image of output celeb3 here
        ))
        cv2.imshow("Analysis Output", combined_display)

    elif key == ord('q'):  # Quit the program when 'q' is pressed
        break

# Release the capture and destroy all OpenCV windows
capture.release()
cv2.destroyAllWindows()
