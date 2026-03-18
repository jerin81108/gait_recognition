import os
import torch
import numpy as np
import random
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

class GaitDataset(Dataset):
    def __init__(self, data_dir, max_frames=60, is_train=True):
        """
        Advanced Gait Dataset for few-shot learning.
        - max_frames: window size for windowed cropping.
        - is_train: if True, applies Random Cropping and Flipping.
        """
        self.data_dir = data_dir
        self.max_frames = max_frames
        self.is_train = is_train
        
        # Gender Mapping: 0=Male, 1=Female
        self.gender_mapping = {
            "person_A": 0,
            "person_B": 1,
            "person_C": 0,
            "person_D": 1
        }
        
        self.data_paths = []
        self.labels = []
        self.gender_labels = []
        
        folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        folders.sort()
        self.person_folders = folders
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.person_folders)
        
        for person in self.person_folders:
            person_dir = os.path.join(data_dir, person)
            label = self.label_encoder.transform([person])[0]
            gender_label = self.gender_mapping.get(person, 0) # Default to Male if unknown
            for file in os.listdir(person_dir):
                if file.endswith('.npy'):
                    self.data_paths.append(os.path.join(person_dir, file))
                    self.labels.append(label)
                    self.gender_labels.append(gender_label)

    def __len__(self):
        # We artificially inflate the dataset size for training to ensure enough updates per epoch
        return len(self.data_paths) * (50 if self.is_train else 1)

    def __getitem__(self, idx):
        # Map inflated index back to real file index
        real_idx = idx % len(self.data_paths)
        file_path = self.data_paths[real_idx]
        label = self.labels[real_idx]
        gender_label = self.gender_labels[real_idx]
        
        # Load features (Frames, 69)
        sequence = np.load(file_path)
        num_frames, feat_dim = sequence.shape
        
        # --- 1. Temporal Augmentation (Random Cropping) ---
        if num_frames > self.max_frames:
            if self.is_train:
                # Pick a random starting point
                start = random.randint(0, num_frames - self.max_frames)
            else:
                # Center crop for validation
                start = (num_frames - self.max_frames) // 2
            sequence = sequence[start : start + self.max_frames, :]
        elif num_frames < self.max_frames:
            # Pad with zeros
            padding = np.zeros((self.max_frames - num_frames, feat_dim))
            sequence = np.vstack((sequence, padding))

        # --- 2. Coordinate Jittering (New for Good-Fit) ---
        if self.is_train:
            # Add tiny Gaussian noise to the first 51 features (pose x,y,conf)
            noise = np.random.normal(0, 0.005, (sequence.shape[0], 51))
            sequence[:, :51] += noise
            
        # --- 3. Horizontal Flipping (Mirroring) ---
        # 50% chance for training
        if self.is_train and random.random() > 0.5:
            # Mirror the X coordinates (features are [x, y, conf])
            # 17 landmarks * 3 = 51 pose features.
            # Pose features are at indices 0-50.
            # x-indices: 0, 3, 6, ..., 48
            for i in range(0, 51, 3):
                # Heuristic: flip x around a center (relative coord flip is easier if normalized)
                # But since these are pixels, we just negate the relative displacement if it were centered.
                # simpler: x_new = 1.0 - x (if normalized 0-1)
                # Assuming pixels, we keep it simple: Mirroring is hard without image width,
                # but we can flip the 'Posture Angle' (index 62) and 'Shoulder Width' remains same.
                sequence[:, i] = -sequence[:, i] 
            
            # Posture angle flip (index 62)
            if feat_dim > 62:
                sequence[:, 62] = -sequence[:, 62]

        sequence = torch.tensor(sequence, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        gender_label = torch.tensor(gender_label, dtype=torch.long)
        
        return sequence, label, gender_label
    
    def get_num_classes(self):
        return len(self.person_folders)
    
    def get_classes(self):
        return self.person_folders

if __name__ == "__main__":
    dataset = GaitDataset(os.path.join('dataset', 'processed'), max_frames=60, is_train=True)
    print(f"Inflated training size: {len(dataset)}")
    path = dataset.data_paths[0]
    data = np.load(path)
    print(f"Sample raw shape: {data.shape}")
    seq, lbl, gnd = dataset[0]
    print(f"Augmented shape: {seq.shape}, Label: {lbl}, Gender: {gnd}")
