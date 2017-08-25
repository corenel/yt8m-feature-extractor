"""Dummy Dataset and data loader for Youtube-8M."""

import os

import skvideo.io
import torch
import torch.utils.data as data
from torchvision import transforms


class VideoFrame(data.Dataset):
    """Dummy dataset for frames in single video."""

    def __init__(self, filepath, num_frames, transform=None):
        """Init VideoFrame dataset."""
        super(VideoFrame, self).__init__()
        self.filepath = filepath
        self.num_frames = num_frames
        self.transform = transform
        self.frames = None
        self.decode()

    def __getitem__(self, index):
        """Get frames from video."""
        frame = self.frames[index, ...]
        if self.transform is not None:
            frame = self.transform(frame)
        return frame

    def __len__(self):
        """Get number of the frames."""
        return self.num_frames

    def decode(self):
        """Decode frames from video."""
        if os.path.exists(self.filepath):
            try:
                self.frames = skvideo.io.vread(
                    self.filepath, num_frames=self.num_frames)
            except AssertionError:
                self.frames = skvideo.io.vread(self.filepath)
            # return numpy.ndarray (N x H x W x C)
            self.frames = skvideo.utils.vshape(self.frames)
            self.num_frames = self.frames.shape[0]
        else:
            print("video file doesn't exist: {}".format(self.filepath))


def get_dataloader(filepath, num_frames, batch_size):
    """Get VideoFrame dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Scale([299, 299]),
                                      transforms.ToTensor()])

    # dataset and data loader
    frame_dataset = VideoFrame(filepath, num_frames, pre_process)

    frame_data_loader = torch.utils.data.DataLoader(
        dataset=frame_dataset,
        batch_size=batch_size,
        shuffle=True)

    return frame_data_loader
