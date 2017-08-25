"""Reader for Youtube-8M dataset."""

import numpy as np
import tensorflow as tf


class Reader(object):
    """Reader for Youtube-8M dataset."""

    def __init__(self, record):
        """Init Reader."""
        super(Reader, self).__init__()
        self.vid = None
        self.labels = None
        self.feat_rgb = None
        self.feat_audio = None

        self.load(record)

    def load(self, record):
        """Load TFRecord."""
        result = tf.train.Example.FromString(record)
        feature = result.features.feature
        self.vid = feature["video_id"].bytes_list.value[0].decode("utf-8")
        self.labels = np.array(list(feature["labels"].int64_list.value))
        self.feat_rgb = np.array(list(feature["mean_rgb"].float_list.value))
        self.feat_audio = np.array(
            list(feature["mean_audio"].float_list.value))
