"""Train and test script for Youtube-8M feature extractor."""

import os

import init_path
import misc.config as cfg
from misc.utils import (concat_feat, concat_feat_var, get_dataloader,
                        make_cuda, make_variable)
from misc.writer import RecordWriter
from models import PCAWrapper, inception_v3

if __name__ == '__main__':
    ############################
    # Init models and datasets #
    ############################
    print("== Init ==")
    # init Inception v3 model
    model = make_cuda(inception_v3(pretrained=True,
                                   transform_input=True,
                                   extract_feat=True))
    model.eval()
    # init PCA model
    pca = PCAWrapper(n_components=cfg.n_components)

    # data loader for frames decoded from several videos
    data_loader_train = get_dataloader(dataset="FrameImage",
                                       path=cfg.frame_root,
                                       batch_size=cfg.batch_size)

    # data loader for frames in ingle video
    data_loader_test = get_dataloader(dataset="VideoFrame",
                                      path=cfg.video_file,
                                      num_frames=cfg.num_frames,
                                      batch_size=cfg.batch_size)

    # init writer
    writer = RecordWriter(filepath=cfg.extract_feat_path, level="frame")

    ###################
    # Train PCA model #
    ###################
    print("== Train ==")
    # extract features by inception_v3
    feats = None
    for step, frames in enumerate(data_loader_train):
        print("extracting feature [{}/{}]".format(step + 1,
                                                  len(data_loader_train)))
        feat = model(make_variable(frames))
        feats = concat_feat_var(feats, feat.data.cpu())

    # train PCA
    X = feats.numpy()
    pca.fit(X)

    #################
    # Test on video #
    #################
    print("== Test ==")
    # extract features by inception_v3
    feats = None
    for step, frames in enumerate(data_loader_test):
        print("extracting feature [{}/{}]".format(step + 1,
                                                  len(data_loader_test)))
        feat = model(make_variable(frames))
        feat_np = feat.data.cpu().numpy()
        # recude dimensions by PCA
        feat_ = pca.transform(feat_np)
        feats = concat_feat(feats, feat_)

    # write features into TFRecord
    vid = os.path.splitext(os.path.basename(cfg.video_file))[0]
    writer.write(vid=vid, feat_rgb=feats)
    writer.close()
