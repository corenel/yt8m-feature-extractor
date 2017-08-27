"""Writer for Youtube-8M dataset."""

import numpy as np
import tensorflow as tf


class RecordWriter(object):
    """RecordWriter for Youtube-8M dataset."""

    def __init__(self, filepath, vid, num_frames, feat_rgb, feat_audio=None):
        """Init RecordWriter."""
        super(RecordWriter, self).__init__()
        self.filepath = filepath
        self.vid = vid
        self.num_frames = num_frames
        self.feat_rgb = feat_rgb
        self.feat_audio = feat_audio
        self.writer = tf.python_io.TFRecordWriter(self.filepath)

    def write(self):
        """Write features into TFRecord."""
        pass

    def save(self):
        """Save TFRecord."""
        pass
