"""Read TFRecord and download corresponding youtube video."""

import argparse

import numpy as np
import pafy
import tensorflow as tf

from reader import Reader


def is_valid(url):
    """Check if the video is available and longer than 3 minutes."""
    pass


def download(filepath):
    """Download videos whose urls are stored in TFRecord from Youtube-8M."""
    pass


def parse():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Read TFRecord and download corresponding youtube video")
    parser.add_argument('filepath',
                        help='path for TFRecord file')
    parser.add_argument("-o", '--save-dir', default='.',
                        help="path to save downloaded videos")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse()
    for record in tf.python_io.tf_record_iterator(args.filepath):
        result = Reader(record)
        url = "https://www.youtube.com/watch?v={}".format(result.vid)
        print(url)
