"""Read TFRecord and download corresponding youtube video."""

import argparse

import numpy as np
import pafy
import tensorflow as tf


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
        result = tf.train.Example.FromString(record)
        v_id = result.features.feature["video_id"].bytes_list.value[0].decode(
            "utf-8")
        tags = list(result.features.feature["labels"].int64_list.value)
        url = "https://www.youtube.com/watch?v={}".format(v_id)
        print(url)
