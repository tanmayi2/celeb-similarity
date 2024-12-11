from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

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
dataset = datasets.ImageFolder('data/demo_images')
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

for i in range(len(average_df.index.tolist())):
    print(f'{faces[i]} looks the most like {top_3_similar_faces[i]} with distances {top_3_similar_faces_dists[i]}') #print out the results

def display_similar_faces(faces, top_3_similar_faces, top_3_similar_faces_dists, image_dir='data/embedding_images'):
    for i, face in enumerate(faces):
        fig, axes = plt.subplots(1, 4, figsize=(16, 6))
        
        # show original face
        input_folder_path = os.path.join("data/demo_images", face)
        input_img_path = os.path.join(input_folder_path, os.listdir(input_folder_path)[0])
        print(input_img_path)
        input_img = Image.open(input_img_path)

        if os.path.exists(input_img_path):
            axes[0].imshow(input_img) 
            axes[0].set_title(f"Original: {face}")
            axes[0].axis('off')
        else:
            print(f"image not found: {input_img_path}")
        
        # show top 3 similar celeb images
        for j, celeb in enumerate(top_3_similar_faces[i]):
            celeb_folder_path = os.path.join(image_dir, celeb)
            celeb_img_path = os.path.join(celeb_folder_path, os.listdir(celeb_folder_path)[0])
            if os.path.exists(celeb_img_path):
                celeb_img = Image.open(celeb_img_path)
                axes[j+1].imshow(celeb_img)
                axes[j+1].set_title(f"{celeb}\nDist: {top_3_similar_faces_dists[i][j]:.2f}")
                axes[j+1].axis('off')
            else:
                print(f"image not found: {celeb_img_path}")
        
        plt.tight_layout()
        plt.show()

# Call the function
display_similar_faces(faces, top_3_similar_faces, top_3_similar_faces_dists, image_dir='data/embedding_images')

