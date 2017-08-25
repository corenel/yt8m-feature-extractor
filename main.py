"""Main script for Youtube-8M feature extractor."""

# import torchvision.models as models
import models
from misc.dataset import get_dataloader
from misc.utils import concat_feat, make_cuda, make_variable

if __name__ == '__main__':
    model = make_cuda(models.inception_v3(pretrained=True,
                                          transform_input=True,
                                          extract_feat=True))
    model.eval()
    data_loader = get_dataloader(filepath="data/1.mp4",
                                 num_frames=360,
                                 batch_size=32)
    features = None
    for frames in data_loader:
        feat = model(make_variable(frames))
        features = concat_feat(features, feat.data.cpu())

    print(features.size())
