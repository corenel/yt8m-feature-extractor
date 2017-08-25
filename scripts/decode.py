"""Decode frames from video."""

import argparse
import os

import skvideo.io


def parse():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Decode frames from video.")
    parser.add_argument('filepath',
                        help='path for video file')
    parser.add_argument("-n", '--num-frames', default=360,
                        help="number of frames to be decoded")
    args = parser.parse_args()

    return args


def decode(filepath, num_frames):
    """Decode frames from video."""
    vid = None
    if os.path.exists(filepath):
        vid = skvideo.io.vread(filepath, num_frames=num_frames)[:, :, :, 0]
        # return numpy.ndarray (N x H x W x C)
        vid = skvideo.utils.vshape(vid)

    return vid


if __name__ == '__main__':
    args = parse()
    decode(args.filepath, args.num_frames)
