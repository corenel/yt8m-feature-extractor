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
    features = None
    for frames in data_loader:
        feat = model(make_variable(frames))
        features = concat_feat(features, feat.data.cpu())

    # recude dimensions by PCA
    pca.fit(features.numpy())
