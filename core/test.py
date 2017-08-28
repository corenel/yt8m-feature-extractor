"""Test script for Youtube-8M feature extractor."""

import os

import misc.config as cfg
from misc.utils import get_dataloader, make_cuda, make_variable
from misc.writer import RecordWriter
from models import PCAWrapper, inception_v3

if __name__ == '__main__':
    # init Inception v3 model
    model = make_cuda(inception_v3(pretrained=True,
                                   transform_input=True,
                                   extract_feat=True))
    model.eval()

    # init PCA model
    pca = PCAWrapper(n_components=cfg.n_components)
    pca.load_params(filepath=cfg.pca_model)

    # data loader for frames in ingle video
    data_loader = get_dataloader(dataset="VideoFrame",
                                 path=cfg.video_file,
                                 num_frames=cfg.num_frames,
                                 batch_size=cfg.batch_size)

    # init writer
    writer = RecordWriter(filepath=cfg.extract_feat_path, level="frame")

    # extract features by inception_v3
    feats = []
    for step, frames in enumerate(data_loader):
        print("extracting feature [{}/{}]".format(step + 1, len(data_loader)))
        feat = model(make_variable(frames))
        # recude dimensions by PCA
        feat_ = pca.transform(feat.data.cpu().numpy())
        feats.append(feat_)

    # write features into TFRecord
    vid = os.path.splitext(os.path.basename(cfg.video_file))[0]
    writer.write(vid=vid, feat_rgb=feats)
    writer.close()
