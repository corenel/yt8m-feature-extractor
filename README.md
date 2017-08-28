# yt8m-feature-extractor
Extract features from video file as the format in Youtube-8M.

## Usage

- `scripts/download.py`: download videos from YouTube corresponding to the TFRecord file.
- `scripts/decode.py`: decode frames from video and save them to data folder.
- `core/train.py`: extract 2048-dims feature from frames of downloaded videos by Inception v3 model and train PCA with them.
- `core/test.py`: extract YouTube-8M-like 1024-dims features from frames of single input video by Inception v3 model and PCA.