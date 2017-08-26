"""Helpful functions."""

import torch
from torch.autograd import Variable
from torchvision import transforms

from datasets import FrameImage, VideoFrame


def make_variable(tensor):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor)


def make_cuda(model):
    """Use CUDA if available."""
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def concat_feat(src, tgt):
    """Concatate features."""
    if src is None:
        out = tgt
    else:
        out = torch.cat([src, tgt])
    return out


def get_dataloader(dataset, path, num_frames=360, batch_size=32):
    """Get dataset loader."""
    # get dataset
    if dataset == "FrameImage":
        pre_process = transforms.Compose([transforms.Scale([299, 299]),
                                          transforms.ToTensor()])
        frame_dataset = FrameImage(path, pre_process)
    elif dataset == "VideoFrame":
        pre_process = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Scale([299, 299]),
                                          transforms.ToTensor()])
        frame_dataset = VideoFrame(path, num_frames, pre_process)

    # get data loader
    frame_data_loader = torch.utils.data.DataLoader(
        dataset=frame_dataset,
        batch_size=batch_size,
        shuffle=True)

    return frame_data_loader
