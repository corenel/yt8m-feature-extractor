"""Config for this project."""

import os

# general
root = "/media/m/E/download/"
# root = "/Volumes/Document/Datasets/download/"

# config for dataset
video_root = os.path.join(root, "video")
frame_root = os.path.join(root, "frame")
record_root = os.path.join(root, "tf")
video_ext = [".mp4"]
video_file = os.path.join("data/videos/005Go8GHXNI.mp4")
num_frames = 300
batch_size = 32

# config for extracting features from Inception v3
save_step = 10000
inception_v3_feats_ext = [".pt"]
inception_v3_feats_root = os.path.join(root, "inception_v3_feats")
inception_v3_feats_path = os.path.join(inception_v3_feats_root, "{}.pt")

# config for PCA
n_components = 1024
pca_batch_size = 4096
pca_model = os.path.join(root, "snapshots", "pca_params.pkl")

# config for extracted features
extract_feat_path = os.path.join(
    root, "extracted_feats", "train-{:0004}.tfrecord")
feats_per_file = 300

# config for labels
csv_root = os.path.join(root, "csv")
vocab_url = "https://research.google.com/youtube8m/csv/vocabulary.csv"
vocab_path = os.path.join(csv_root, "vocabulary.csv")
pred_path = os.path.join(csv_root, "pred.csv")
top_k = 5
