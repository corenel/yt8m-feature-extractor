"""Check if TFRecord is valid and complete."""

import argparse
import os

import tensorflow as tf

from misc.reader import Reader


def parse():
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Read TFRecord and check its validation")
    parser.add_argument('filepath',
                        help='path for TFRecord file')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse()
    for record_file in os.listdir(args.filepath):
        print("checking record file: {}".format(record_file))
        record_iterator = tf.python_io.tf_record_iterator(
            os.path.join(args.filepath, record_file))
        for idx, record in enumerate(record_iterator):
            result = Reader(record)
            print("--> checking {} [{}/{}]".format(result.vid,
                                                   idx + 1,
                                                   len(record_iterator)))
    print("All checked!")
