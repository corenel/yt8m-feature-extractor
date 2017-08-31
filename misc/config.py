"""Config for this project."""

# config for dataset
video_root = "data/videos"
frame_root = "data/frames"
video_file = "data/videos/005Go8GHXNI.mp4"
num_frames = 300
batch_size = 32

# config for PCA
n_components = 1024
pca_model = "snapshots/pca_params.pkl"

# config for extracted features
extract_feat_path = "data/extracted.tfrecord"
