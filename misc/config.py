"""Config for this project."""

# config for dataset
video_ext = [".mp4"]
video_root = "data/videos"
frame_root = "data/frames"
video_file = "data/videos/005Go8GHXNI.mp4"
num_frames = 300
batch_size = 32

# config for extracting features from Inception v3
save_step = 100
feats_path = "snapshots/inception_v3_feats-{}.pt"

# config for PCA
n_components = 1024
pca_model = "snapshots/pca_params.pkl"

# config for extracted features
extract_feat_path = "data/extracted.tfrecord"
feats_per_file = 100
