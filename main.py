"""Main script for Youtube-8M feature extractor."""

import misc.config as cfg
from misc.dataset import get_dataloader
from misc.utils import concat_feat, make_cuda, make_variable
from models import PCAWrapper, inception_v3

if __name__ == '__main__':
    # init models and data loader
    model = make_cuda(inception_v3(pretrained=True,
                                   transform_input=True,
                                   extract_feat=True))
    pca = PCAWrapper(n_components=cfg.n_components)
    model.eval()
    data_loader = get_dataloader(filepath=cfg.filepath,
                                 num_frames=cfg.num_frames,
                                 batch_size=cfg.batch_size)

    # extract features by inception_v3
    feats = None
    for step, frames in enumerate(data_loader):
        print("extracting feature [{}/{}]".format(step + 1, len(data_loader)))
        feat = model(make_variable(frames))
        feats = concat_feat(feats, feat.data.cpu())

    # recude dimensions by PCA
    X = feats.numpy()
    print(X.shape)
    pca.fit(X)
    X_ = feats[0, :].numpy().reshape(1, -1)
    print(X_.shape)
    print(pca.transform(X_).shape)
