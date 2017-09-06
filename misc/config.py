"""Config for this project."""

import os

# general
root = "/media/m/F1/download/"

# config for dataset
video_ext = [".mp4"]
video_root = os.path.join(root, "video")
frame_root = os.path.join(root, "frame")
record_root = os.path.join(root, "tf")
video_file = os.path.join(video_root, "005Go8GHXNI.mp4")
num_frames = 300
batch_size = 32

# config for extracting features from Inception v3
save_step = 10000
inception_v3_feats_ext = [".pt"]
inception_v3_feats_root = os.path.join(root, "inception_v3_feats")
inception_v3_feats_path = os.path.join(inception_v3_feats_root, "{}.pt")

# config for PCA
n_components = 1024
pca_model = os.path.join(root, "snapshots", "pca_params.pkl")

# config for extracted features
extract_feat_path = os.path.join(root, "extracted_feats", "train-{}.pt")
feats_per_file = 100
