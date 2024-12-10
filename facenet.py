from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity

workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)), #might want to remove this
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class CelebADataset(Dataset):
    def __init__(self, img_dir, identity_file, bbox_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = pd.read_csv(identity_file, sep=' ', header=None, names=['image', 'identity'])
        self.image_paths = self.data['image'].values
        self.identities = self.data['identity'].values
        self.bbox_data = pd.read_csv(bbox_file)

        self.bbox_dict = {
            row['image_id']: (row['x_1'], row['y_1'], row['width'], row['height'])
            for _, row in self.bbox_data.iterrows()
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        identity = self.identities[idx]

        #bounding box
        bbox = self.bbox_dict.get(self.image_paths[idx])
        if bbox:
            x, y, w, h = bbox
            image = image.crop((x, y, x + w, y + h))

        if self.transform:
            image = self.transform(image)
        return image, identity

#dataloader
project_root = os.path.dirname(__file__) 

#put celeba dataset under data directory
img_dir = os.path.join(project_root, 'data', 'celeba', 'img_align_celeba')
identity_file = os.path.join(project_root, 'data', 'celeba', 'identity_CelebA.txt')
bbox_file = os.path.join(project_root, 'data', 'celeba', 'list_bbox_celeba.csv')
dataset = CelebADataset(img_dir, identity_file, bbox_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=workers)

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    return x[0]


aligned = []
ids = []
for images, identities in dataloader:
    images = images.to(device)
    x_aligned, probs = mtcnn(images, return_prob=True)
    if x_aligned is not None:
        for aligned_img, prob, identity in zip(x_aligned, probs, identities):
            if prob is not None and prob > 0.9:  # Confidence threshold
                aligned.append(aligned_img)
                ids.append(identity.item())

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

# create dataframe with the identities and embeddings
df = pd.DataFrame({"identity": ids, "embedding": list(embeddings)})

# group by identity and average embeddings
def average_embeddings(embeddings):
    return np.mean(np.vstack(embeddings), axis=0)

average_df = df.groupby('identity')['embedding'].agg(average_embeddings)

#create distribution chart
dists = [[torch.tensor(e1 - e2).norm().item() for e2 in average_df] for e1 in average_df]
print(pd.DataFrame(dists, columns=average_df.index.to_list(), index=average_df.index.to_list()))

#compute top 3 similarities
def get_top_similarities(target_embedding, average_df):
    embeddings_matrix = np.vstack(average_df.values)
    similarities = cosine_similarity(target_embedding.reshape(1, -1), embeddings_matrix)[0]
    top_indeces = similarities.argsort()[-3:][::-1]
    top_3 = [(average_df.index[i], similarities[i]) for i in top_indeces]
    return top_3

