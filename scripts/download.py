"""Read TFRecord and download corresponding youtube video."""

import argparse
import os

import pafy
import tensorflow as tf

from misc.reader import Reader


def is_valid(video):
    """Check if the video is available and longer than 3 minutes."""
    return video.length != -1


def download(video, save_dir, vid):
    """Download videos whose urls are stored in TFRecord from Youtube-8M."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print("downloading {}".format(video.title))
    best = video.getbest(preftype="mp4")
    filename = best.download(
        filepath=os.path.join(save_dir, "{}.{}".format(vid, best.extension)))
    print("saved to {}".format(filename))


def parse():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Read TFRecord and download corresponding youtube video")
    parser.add_argument('filepath',
                        help='path for TFRecord file')
    parser.add_argument("-o", '--save-dir', default='data/videos/',
                        help="path to save downloaded videos")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse()
    for record in tf.python_io.tf_record_iterator(args.filepath):
        result = Reader(record)
        url = "https://www.youtube.com/watch?v={}".format(result.vid)
        video = pafy.new(url)
        if is_valid(video):
            download(video, args.save_dir, result.vid)
