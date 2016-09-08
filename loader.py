
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import glob
import threading
import string


import numpy as np
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class JpegCoder(object):
    def __init__(self):
        self.sess = tf.Session()
        
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

        self.decode_jpeg_data = tf.placeholder(dtype = tf.string)
        self.decode_jpeg = tf.image.decode_jpeg(self.decode_jpeg_data, channels = 3)
        
    def decoder_jpeg(self, image_data):
        image = self.sess.run(self.decode_jpeg, feed_dict = {self.decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image
    
    def cmyk_to_rgb(self, image_data):
        return self.sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

def _is_cmyk(filename):
    """Determine if file contains a CMYK JPEG format image.
    Args:
    filename: string, path of the image file.
    Returns:
    boolean indicating if the image is a JPEG encoded with CMYK color space.
    """
    # File list from:
    # https://github.com/cytsai/ilsvrc-cmyk-image-list
    blacklist = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
               'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
               'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
               'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
               'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
               'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
               'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
               'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
               'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
               'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
               'n07583066_647.JPEG', 'n13037406_4650.JPEG']
    return filename.split('/')[-1] in blacklist

def process_image(filename, coder):
    image_data = tf.gfile.FastGFile(filename, 'rb').read()
    
    if _is_cmyk(filename):
        image_data = coder.cmyk_to_rgb(image_data)
    
    image = coder.decoder_jpeg(image_data)
    
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] ==3
    
    return image_data, height, width

def process_image_batch(coder, name, thread_index, ranges, synsets, filenames, labels, num_shards):
    num_threads = len(ranges)
    assert not num_shards % num_threads
    shards_in_batch = int(num_shards / num_threads)
    
    shard_ranges = np.linspace(ranges[thread_index][0], 
                               ranges[thread_index][1], shards_in_batch + 1).astype(int)
    num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]
    
    counter = 0
    for i in range(shards_in_batch):
        shard = thread_index * shards_in_batch + i
        output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
        output_file = os.path.join('/data/ImageNet/output_data/', output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)
        
        shard_counter = 0
        files_in_shard = np.arange(shard_ranges[i], shard_ranges[i+1], dtype=int)
        for j in files_in_shard:

            filename = filenames[j]
            label = labels[j]
            synset = synsets[j]
            
            image_buffer, height, width = process_image(filename, coder)
            
            colorspace = b'RGB'
            channels = 3
            image_format = b'JPEG'
            basename = str.encode(os.path.basename(filename))
            synset = str.encode(synset)
            
            example = tf.train.Example(features = tf.train.Features(feature = {
                'image/height': _int64_feature(height),
                'image/width': _int64_feature(width),
                'image/colorspace': _bytes_feature(colorspace),
                'image/channels': _int64_feature(channels),
                'image/class/label': _int64_feature(label),
                'image/class/synsets': _bytes_feature(synset),
                'image/format': _bytes_feature(image_format),
                'image/filename': _bytes_feature(basename),
                'image/encoded': _bytes_feature(image_buffer)}))
                        
              
            writer.write(example.SerializeToString())
            shard_counter += 1
            counter += 1
            if not counter % 1000:
                sys.stdout.flush()
        sys.stdout.flush()    
        shard_counter = 0
    sys.stdout.flush()
      
def get_file_data(data_dir, chal_synsets):
    label_index = 1
    labels = []
    synsets  = []
    filenames = []

    for synset in chal_synsets:
        synset = synset.strip()
        jpeg_path = data_dir + '%s/*.JPEG'  % synset
        matching_files = glob.glob(jpeg_path)

        labels.extend([label_index] * len(matching_files))
        synsets.extend([synset] * len(matching_files))
        filenames.extend(matching_files)

        label_index += 1

    shuffled_index = range(len(filenames))
    random.seed(12345)
    random.shuffle(filenames)    

    filenames = [filenames[i] for i in shuffled_index]
    synsets = [synsets[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]
    
    return filenames, synsets, labels      
            
# eval_labels = [l.strip() for l in open('/data/ImageNet/dev_kit/imagenet_2012_validation_synset_labels.txt')
#               .readlines()]
chal_synsets = []
enc = 'utf-8'

f = open('/data/ImageNet/dev_kit/traindata_labels.txt', 'r')

chal_synsets = f.readlines()



data_dir_eval = '/data/ImageNet/eval_data/'
data_dir_train = '/data/ImageNet/train_data/'

filenames, synsets, labels = get_file_data(data_dir_train, chal_synsets)
eval_filenames, eval_synsets, eval_labels = get_file_data(data_dir_eval, chal_synsets)

"""train_shards = 1024
eval_shards = 128
num_threads = 16

spacing = np.linspace(0, len(filenames), num_threads + 1).astype(np.int)
ranges = []
threads = []
for j in range(len(spacing) - 1):
    ranges.append([spacing[j], spacing[j+1]])

# launch thread
sys.stdout.flush()

# monitoring
coord = tf.train.Coordinator()

coder = JpegCoder()

threads = []
for thread_index in range(len(ranges)):
    args = (coder, 'train', thread_index, ranges, synsets, filenames, labels, train_shards)
    t = threading.Thread(target = process_image_batch, args=args)
    t.start()
    threads.append(t)
    
    coord.join(threads)
    sys.stdout.flush()

spacing = np.linspace(0, len(eval_filenames), num_threads + 1).astype(np.int)
ranges = []
threads = []
for j in range(len(spacing) - 1):
    ranges.append([spacing[j], spacing[j+1]])

# launch thread
sys.stdout.flush()

# monitoring
coord = tf.train.Coordinator()

coder = JpegCoder()

threads = []
for thread_index in range(len(ranges)):
    args = (coder, 'eval', thread_index, ranges, eval_synsets, eval_filenames, eval_labels, eval_shards)
    t = threading.Thread(target = process_image_batch, args=args)
    t.start()
    threads.append(t)
    
    coord.join(threads)
    sys.stdout.flush()"""

from abc import ABCMeta
from abc import abstractmethod

class Dataset(object):
    """A simple class for handling data sets."""
    __metaclass__ = ABCMeta

    def __init__(self,  subset):
        """Initialize dataset using a subset and the path to the data."""
        assert subset in self.available_subsets(), self.available_subsets()
        self.subset = subset

    @abstractmethod
    def num_classes(self):
        return 1000

    @abstractmethod
    def num_examples_per_epoch(self):
        """Returns the number of examples in the data subset."""
        if self.subset == 'train':
            return 1281167
        if self.subset == 'eval':
            return 50000

    @abstractmethod


    def available_subsets(self):
        """Returns the list of available subsets."""
        return ['train', 'eval']

    def data_files(self):
        """Returns a python list of all (sharded) data subset files.
        Returns:
          python list of all (sharded) data set files.
        Raises:
          ValueError: if there are not data_files matching the subset.
        """
        tf_record_pattern = os.path.join('/data/ImageNet/output_data/', '%s-*' % self.subset)
        data_files = tf.gfile.Glob(tf_record_pattern)
        if not data_files:
            print('No files found for dataset %s' % (self.subset))

            exit(-1)
        return data_files

    def reader(self):
        """Return a reader for a single entry from the data set.
        See io_ops.py for details of Reader class.
        Returns:
          Reader object that reads the data set.
        """
        return tf.TFRecordReader()

def jpeg_to_tensor(image_buffer, scope=None):
    """Decode a JPEG string into one 3-D float image Tensor.
    Args:
    image_buffer: scalar string Tensor.
    scope: Optional scope for op_scope.
    Returns:
    3-D float Tensor with values ranging from [0, 1).
    """
    with tf.op_scope([image_buffer], scope, 'decode_jpeg'):
        # Decode the string as an RGB JPEG.
        # Note that the resulting image contains an unknown height and width
        # that is set dynamically by decode_jpeg. In other words, the height
        # and width of image is unknown at compile-time.
        image = tf.image.decode_jpeg(image_buffer, channels=3)

        # After this point, all image pixels reside in [0,1)
        # until the very end, when they're rescaled to (-1, 1).  The various
        # adjust_* ops all require this range for dtype float.
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image


def distort_color(image, thread_id=0, scope=None):
    """Distort the color of the image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
    image: Tensor containing single image.
    thread_id: preprocessing thread ID.
    scope: Optional scope for op_scope.
    Returns:
    color-distorted image
    """
    with tf.op_scope([image], scope, 'distort_color'):
        color_ordering = thread_id % 2

        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        # The random_* ops do not necessarily clamp.
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def distort_image(image, height, width, thread_id=0, scope=None):
    """Distort one image for training a network.
    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.
    Args:
    image: 3-D float Tensor of image
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    thread_id: integer indicating the preprocessing thread.
    scope: Optional scope for op_scope.
    Returns:
    3-D float Tensor of distorted image used for training.
    """
    distorted_image = image

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.
    resize_method = thread_id % 4
    distorted_image = tf.image.resize_images(distorted_image, height, width,
                                             resize_method)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([height, width, 3])
    if not thread_id:
        tf.image_summary('cropped_resized_image',
                       tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors.
    distorted_image = distort_color(distorted_image, thread_id)

    if not thread_id:
        tf.image_summary('final_distorted_image',
                       tf.expand_dims(distorted_image, 0))
    return distorted_image


def eval_image(image, height, width, scope=None):
    """Prepare one image for evaluation.
    Args:
    image: 3-D float Tensor
    height: integer
    width: integer
    scope: Optional scope for op_scope.
    Returns:
    3-D float Tensor of prepared image.
    """
    with tf.op_scope([image, height, width], scope, 'eval_image'):
        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        image = tf.image.central_crop(image, central_fraction=0.875)

        # Resize the image to the original height and width.
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bilinear(image, [height, width],
                                         align_corners=False)
        image = tf.squeeze(image, [0])
        return image


def image_preprocessing(image_buffer, train, thread_id=0):
    """Decode and preprocess one image for evaluation or training.
    Args:
    image_buffer: JPEG encoded string Tensor
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    train: boolean
    thread_id: integer indicating preprocessing thread
    Returns:
    3-D float Tensor containing an appropriately scaled image
    Raises:
    ValueError: if user does not provide bounding box
    """
    
    image = jpeg_to_tensor(image_buffer)
    height = 224
    width = 224

    if train:
        image = distort_image(image, height, width, thread_id)
    else:
        image = eval_image(image, height, width)

    # Finally, rescale to [-1,1] instead of [0, 1)
    image = tf.sub(image, 0.5)
    image = tf.mul(image, 2.0)
    return image


def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.
    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:
    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615


    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>
    Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([1], dtype=tf.int64,
                                              default_value=-1),
      'image/class/synset': tf.FixedLenFeature([], dtype=tf.string, default_value='')

    }
    sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.

    features = tf.parse_single_example(example_serialized, feature_map)
    label = tf.cast(features['image/class/label'], dtype=tf.int32)



    return features['image/encoded'], label, features['image/class/synset']

def batch_inputs(dataset, batch_size, train, num_preprocess_threads, num_readers):
    data_files = dataset.data_files()
    if data_files is None:
        raise ValueError('No data files found for this dataset')

    # Create filename_queue
    if train:
        filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=True,
                                                      capacity=16)
    else:
        filename_queue = tf.train.string_input_producer(data_files,
                                                      shuffle=False,
                                                      capacity=1)
    if num_preprocess_threads is None:
        num_preprocess_threads = FLAGS.num_preprocess_threads

    if num_preprocess_threads % 4:
        raise ValueError('Please make num_preprocess_threads a multiple '
                       'of 4 (%d % 4 != 0).', num_preprocess_threads)

    if num_readers is None:
        num_readers = FLAGS.num_readers

    if num_readers < 1:
        raise ValueError('Please make num_readers at least 1')
        
    examples_per_shard = 1024
    
    min_queue_examples = examples_per_shard * 16
    if train:
        examples_queue = tf.RandomShuffleQueue(
              capacity=min_queue_examples + 3 * batch_size,
              min_after_dequeue=min_queue_examples,
              dtypes=[tf.string])
    else:
        examples_queue = tf.FIFOQueue(
              capacity=examples_per_shard + 3 * batch_size,
              dtypes=[tf.string])

    # Create multiple readers to populate the queue of examples.
    if num_readers > 1:
        enqueue_ops = []
        for _ in range(num_readers):
            reader = dataset.reader()
            _, value = reader.read(filename_queue)
            enqueue_ops.append(examples_queue.enqueue([value]))

            tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
            example_serialized = examples_queue.dequeue()
    else:
        reader = dataset.reader()
        _, example_serialized = reader.read(filename_queue)

    images_and_labels = []
    for thread_id in range(num_preprocess_threads):
        # Parse a serialized Example proto to extract the image and metadata.
        image_buffer, label_index, _ = parse_example_proto(
          example_serialized)
        image = image_preprocessing(image_buffer, train, thread_id)
        images_and_labels.append([image, label_index])

        images, label_index_batch = tf.train.batch_join(
                images_and_labels,
                batch_size=batch_size,
                capacity=2 * num_preprocess_threads * batch_size)

    # Reshape images into these desired dimensions.
    height = 224
    width = 224
    depth = 3

    images = tf.cast(images, tf.float32)
    images = tf.reshape(images, shape=[batch_size, height, width, depth])

    # Display the training images in the visualizer.
    tf.image_summary('images', images)

    return images, tf.reshape(label_index_batch, [batch_size])

train_dataset = Dataset('train')
eval_dataset = Dataset('eval')
batch_size = 128
num_preprocess_threads = 4

g = tf.Graph()

with g.as_default():
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    with sess.as_default():
        with tf.device('cpu:0'):
            train_images, train_labels = batch_inputs(train_dataset, batch_size, 
                                                      True, num_preprocess_threads, 4)
            eval_images, eval_labels = batch_inputs(eval_dataset, batch_size, 
                                                    False, num_preprocess_threads, 1)

            sess.run(tf.initialize_all_variables())
            tf.train.start_queue_runners(sess)

        print(eval_labels.eval())


































