"""Config for this project."""

# config for dataset
video_root = "data/videos"
frame_root = "data/frames"
video_file = "data/videos/00_5y0D1Y1w.mp4"
num_frames = 360
batch_size = 32

# config for PCA
n_components = 1024
pca_model = "snapshots/pca_params.pkl"

# config for extracted features
extract_feat_path = "data/extracted.tfrecord"
