import cv2
import os
import numpy as np

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image




# Placeholder function for processing the image
def process_image(image):
    tensor1 = torch.load("data/embeddings/embeddings.pt")
    tensor2 = torch.load("data/embeddings/bollywood_embeddings2.pt")

    celeb_embeddings = pd.concat([tensor1, tensor2])
    celeb_names = celeb_embeddings.index.tolist() 

    # print(celeb_embeddings.tail())
    # print(len(celeb_embeddings))


    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True
    )

    def collate_fn(x):
        return x[0]


    #embed the test images
    dataset = datasets.ImageFolder('data/captured_frames')
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn)

    aligned = []
    names = []
    counter = 0
    for x, y in loader:
        x_aligned, prob = mtcnn(x, return_prob=True)
        if x_aligned is not None:
            aligned.append(x_aligned)
            names.append(dataset.idx_to_class[y])

    aligned = torch.stack(aligned)
    embeddings = resnet(aligned).detach()

    #create average embeddings for each person in the test set
    df = pd.DataFrame({"name": names, "embedding": list(embeddings)})

    def average_embeddings(embeddings):
        return np.mean(np.vstack(embeddings), axis=0)


    average_df = df.groupby('name')['embedding'].agg(average_embeddings)
    faces = average_df.index.tolist()

    # find who each person in the test folder looks the most alike
    dists = [[torch.tensor(e1 - e2).norm().item() for e2 in celeb_embeddings] for e1 in average_df]
    top_3_indices = [sorted(range(len(row)), key=lambda i: row[i]) for row in dists]
    # retrieve the top 3 most similar celebs for each person
    top_3_similar_faces = [[celeb_names[i] for i in indices[:3]] for indices in top_3_indices]
    top_3_similar_faces_dists = [[row[i] for i in indices[:3]] for row, indices in zip(dists, top_3_indices)]

    # for i in range(len(average_df.index.tolist())):
    #     print(f'{faces[i]} looks the most like {top_3_similar_faces[i]} with distances {top_3_similar_faces_dists[i]}') #print out the results
    image_dir = 'data/embedding_images'
    for i, face in enumerate(faces):
        out = []
        for j, celeb in enumerate(top_3_similar_faces[i]):
                celeb_folder_path = os.path.join(image_dir, celeb)
                celeb_img_path = os.path.join(celeb_folder_path, os.listdir(celeb_folder_path)[0])
                
                if os.path.exists(celeb_img_path):
                    celeb_img = Image.open(celeb_img_path)
                    out.append(celeb_img)
                else:
                    print(f"image not found: {celeb_img_path}")
        print("output:  ",out)    
        return out[0], out[1], out[2]

# Create a directory to store the frames
output_dir = "data/captured_frames/guest"
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
        celeb1_np = np.array(celeb1)
        celeb2_np = np.array(celeb2)
        celeb3_np = np.array(celeb3)

        frame_height, frame_width, _ = frame.shape



        # Display all four images
        combined_display = np.hstack((
            frame,  # Original
            cv2.cvtColor(cv2.resize(np.array(celeb1), (frame_width, frame_height)), cv2.COLOR_RGB2BGR),  # image of output celeb1 here
            cv2.cvtColor(cv2.resize(np.array(celeb2), (frame_width, frame_height)), cv2.COLOR_RGB2BGR),  # image of output celeb2 here
            cv2.cvtColor(cv2.resize(np.array(celeb3), (frame_width, frame_height)), cv2.COLOR_RGB2BGR)    # image of output celeb3 here
        ))
        cv2.imshow("Analysis Output", combined_display)

    elif key == ord('q'):  # Quit the program when 'q' is pressed
        break

# Release the capture and destroy all OpenCV windows
capture.release()
cv2.destroyAllWindows()
