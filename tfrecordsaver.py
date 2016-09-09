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

def create_records():
	chal_synsets = []
	enc = 'utf-8'

	f = open('/data/ImageNet/dev_kit/traindata_labels.txt', 'r')

	chal_synsets = f.readlines()



	data_dir_eval = '/data/ImageNet/eval_data/'
	data_dir_train = '/data/ImageNet/train_data/'

	filenames, synsets, labels = get_file_data(data_dir_train, chal_synsets)
	eval_filenames, eval_synsets, eval_labels = get_file_data(data_dir_eval, chal_synsets)	
	train_shards = 1024
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
		sys.stdout.flush()









  
            
