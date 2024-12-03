# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import mediapipe as mp

facial_expression = {
    'Angry' :     0,
    'Happy' :     1,
    'Neutral' :   2,
    'Sad' :       3,
    'Surprised' : 4,
    'Fear' :      5,
    'Disgust' :   6
}

def split_dataset(train_dir, test_dir, val_ratio=0.2, seed=42):
    random.seed(seed)

    def extract_video_paths_and_labels(data_dir):

        video_paths, video_labels = [], []

        for label_name in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label_name)
            if not os.path.isdir(label_dir):
                continue

            video_to_frames = {}
            for frame_name in sorted(os.listdir(label_dir)):
                frame_name = frame_name.strip()
                video_id = '_'.join(frame_name.split('_')[:-1])
                frame_path = os.path.join(label_dir, frame_name)

                if video_id not in video_to_frames:
                    video_to_frames[video_id] = []
                video_to_frames[video_id].append(frame_path)

            for frames in video_to_frames.values():
                video_paths.append(frames)
                video_labels.append(facial_expression.get(label_name, -1))

        return video_paths, video_labels


    train_video_paths, train_video_labels = extract_video_paths_and_labels(train_dir)
    test_paths, test_labels = extract_video_paths_and_labels(test_dir)


    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_video_paths, train_video_labels, test_size=val_ratio, random_state=seed
    )

    return train_paths, val_paths, test_paths, train_labels, val_labels, test_labels

def video_to_frames(video_path, output_dir, fps=5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open the video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
            cv2.imwrite(frame_name, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    # print(f"Saved {saved_count} frames to: {output_dir}")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def process_image(input_path, max_frames=16, resize=(112, 112)):
    """
    Processing videos or images, converting them into fixed-size frame sequences, and returning a normalized numpy array.
    Args:
        input_path
        max_frames
        resize: resize to (width, height)
    Returns:
        numpy array, shape (seq_len, height, width, channels)
    """
    def detect_face_and_landmarks(frame):
        """Detects faces and overlays facial landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Draw facial landmarks
        return frame

    frames = []

    if isinstance(input_path, list):
        for path in input_path:
            if isinstance(path, str) and os.path.isfile(path):
                # print(f"Reading file: {path}")
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    frame = cv2.imread(path)
                    if frame is not None:
                        # 確保影像是 RGB 格式
                        if len(frame.shape) < 3 or frame.shape[2] == 1:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        frame = cv2.resize(frame, resize)
                        frame = detect_face_and_landmarks(frame)
                        frame = frame.astype(np.float32) / 255.0
                        frames = [frame] * max_frames
                    else:
                        print(f"Warning: Failed to read {path}")
                else:
                    print(f"Warning: Unsupported file format for {path}")
            else:
                print(f"Warning: Invalid file path {path}")

    elif isinstance(input_path, str) and os.path.isfile(input_path):
        # print(f"Reading file: {input_path}")
        if input_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            frame = cv2.imread(input_path)
            if frame is not None:
                frame = cv2.resize(frame, resize)
                frame = detect_face_and_landmarks(frame)
                frame = frame.astype(np.float32) / 255.0
                frames = [frame] * max_frames
        else:
            print(f"Warning: Unsupported video format for {input_path}")

    return np.array(frames)  # shape: (seq_len, height, width, channels)

def augment_frames(frames):
    """
    Args:
        frames: shape (seq_len, height, width, channels)
    Returns:
        augmented frames
    """
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    augmented_frames = []

    for frame in frames:
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame
        augmented_frame = transform(frame)
        augmented_frames.append(augmented_frame)

    # print(f"Shape after augment_frames: {np.array(augmented_frames).shape}")
    return torch.stack(augmented_frames)  # shape: (seq_len, height, width, channels)

class ResNet3DModel(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3DModel, self).__init__()
        self.backbone = r3d_18(pretrained = True)

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class VideoImageDataset(Dataset):
    def __init__(self, video_paths, labels=None, transform=None, clip_length=16, resize=(112, 112)):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.clip_length = clip_length
        self.resize = resize

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        frame_paths = self.video_paths[idx]  # get all videos' paths
        label = self.labels[idx]  # get the coresponding labels

        frames = self._load_and_process_frames(frame_paths)

        if self.transform:
            frames = self.transform(frames)
            frames = np.transpose(frames, (1, 0, 2, 3))
        else:
            frames = np.transpose(frames, (3, 1, 2, 0))

        frames = torch.tensor(frames, dtype=torch.float32)
        # frames = frames.unsqueeze(0)

        label = torch.tensor(label, dtype=torch.long)

        # print(f"Shape after torch.transpose: {np.array(frames).shape}")
        return frames, label

    def _load_and_process_frames(self, frame_paths):
        """
        Returns:
            np.ndarray: shape = (clip_length, height, width, channels)
        """
        frames = process_image(frame_paths, max_frames=self.clip_length, resize=self.resize)

        # print(f"Shape after _load_and_process_frames: {np.array(frames).shape}")
        return frames

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs = torch.stack([torch.tensor(input) for input in inputs], dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    # print("inputs' shape after collate_fn: ", inputs.shape)
    # print("labels' shape after collate_fn: ", labels.shape)
    return inputs, labels