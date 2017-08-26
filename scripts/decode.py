"""Decode frames from video."""

import argparse
import os

from scipy.misc import imsave

import skvideo.io


def parse():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Decode frames from video.")
    parser.add_argument('filepath',
                        help='path for video folder')
    parser.add_argument("-n", '--num-frames', default=360,
                        help="number of frames to be decoded")
    parser.add_argument("-o", '--save-dir', default='data/frames/',
                        help="path to save downloaded videos")
    args = parser.parse_args()

    return args


def decode(filepath, num_frames, save_dir):
    """Decode frames from video."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists(filepath):
        try:
            videogen = skvideo.io.vreader(filepath, num_frames=num_frames)
        except AssertionError:
            videogen = skvideo.io.vreader(filepath)

        filename = os.path.splitext(os.path.basename(filepath))[0]

        for idx, frame in enumerate(videogen):
            print("decoding {}-{}".format(filename, idx))
            imsave(os.path.join(save_dir, "{}-{}.jpg".format(filename, idx)),
                   frame)


if __name__ == '__main__':
    args = parse()
    for video_file in os.listdir(args.filepath):
        if os.path.splitext(video_file)[1] == ".mp4":
            decode(os.path.join(args.filepath, video_file),
                   args.num_frames,
                   args.save_dir)
