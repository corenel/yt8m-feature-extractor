"""Convert tfrecord to plain text file."""

import argparse
import os

import tensorflow as tf

import init_path
from misc.reader import Reader


def parse():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Read TFRecord and check its validation")
    parser.add_argument("filepath", help="folder path for TFRecord file")
    parser.add_argument(
        "-o", "--output", type=str, default="tfrecord.txt", help="output path")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # parse args
    args = parse()

    # create blank output
    if os.path.exists(os.path.dirname(args.output)):
        os.makedirs(os.path.dirname(args.output))

    # convert record
    record_list = [
        r for r in os.listdir(args.filepath)
        if os.path.splitext(r)[1] == ".tfrecord"
    ]
    with open(args.output, "w") as f:
        for idx, record_file in enumerate(record_list):
            print("checking record file: {} [{}/{}]".format(
                record_file, idx + 1, len(os.listdir(args.filepath))))
            record_iterator = tf.python_io.tf_record_iterator(
                os.path.join(args.filepath, record_file))
            for record in record_iterator:
                result = Reader(record)
                print("--> checking {}".format(result.vid))
                f.write("{}#{}#{}\n".format(result.vid, result.labels,
                                            result.feat_rgb))
    print("All checked!")
