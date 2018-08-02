import glob
import random
import struct
import sys
import tensorflow as tf

class DataReader(object):
    def example_generator(data_path, num_epochs=None):
    """Generates tf.Examples from path of data files.

        Binary data format: <length><blob>. <length> represents the byte size
        of <blob>. <blob> is serialized tf.Example proto. The tf.Example contains
        the tokenized article text and summary.

      Args:
        data_path: path to tf.Example data files.
        num_epochs: Number of times to go through the data. None means infinite.

      Yields:
        Deserialized tf.Example.

      If there are multiple files specified, they accessed in a random order.
      """
      epoch = 0
      while True:
        if num_epochs is not None and epoch >= num_epochs:
          break
        filelist = glob.glob(data_path)
        assert filelist, 'Empty filelist.'
        random.shuffle(filelist)
        for f in filelist:
          reader = open(f, 'rb')
          while True:
            len_bytes = reader.read(8)
            if not len_bytes: break
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            yield tf.train.Example.FromString(example_str)

        epoch += 1

