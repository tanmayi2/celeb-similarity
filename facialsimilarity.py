from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

#load in the embeddings of celebrities
celeb_embeddings = torch.load("../data/embeddings/embeddings.pt") 
celeb_names = celeb_embeddings.index.tolist() 

#load in the model
workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    return x[0]

#embed the test images
dataset = datasets.ImageFolder('../data/test_images')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
names = []
counter = 0
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

#create average embeddings for each person in the test set
df = pd.DataFrame({"name": names, "embedding": list(embeddings)})

def average_embeddings(embeddings):
    return np.mean(np.vstack(embeddings), axis=0)
average_df = df.groupby('name')['embedding'].agg(average_embeddings)
faces = average_df.index.tolist()

# find who each person in the test folder looks the most alike
dists = [[torch.tensor(e1 - e2).norm().item() for e2 in celeb_embeddings] for e1 in average_df]
indicies = [row.index(min(row)) for row in dists]
similar_faces = [celeb_names[i] for i in indicies]

for i in range(len(average_df.index.tolist())):
    print(f'{faces[i]} looks the most like {similar_faces[i]}') #print out the results


