import tensorflow as tf
import numpy as np



num_targets = 1000
random = np.random.randn(1, 224,224,3)
random_target = np.random.randn(1, num_targets)

def model(input_data, targets):
	
	sess = tf.Session()	

	# first convolution layer
	W_conv1 = tf.Variable(tf.truncated_normal([11, 11, 3, 48], stddev=0.1))
	b_conv1 = tf.Variable(tf.constant(0.1, shape = [48]))

	x_image = tf.reshape(input_data, [-1, 224, 224, 3])

	# hidden layer with ReLU and max pooling
	h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, [1, 4, 4, 1], padding = 'SAME') + b_conv1)

	# second convolution layer
	W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 48, 128], stddev=0.1))
	b_conv2 = tf.Variable(tf.constant(0.1, shape = [128]))

	# hidden layer with ReLU and max pooling
	h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, [1, 1, 1, 1], padding = 'SAME') + b_conv2)
	h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding = 'VALID')

	# third convolution layer
	W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 128, 192], stddev=0.1))
	b_conv3 = tf.Variable(tf.constant(0.1, shape = [192]))

	# hidden layer with ReLU
	h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, [1, 1, 1, 1], padding = 'SAME') + b_conv3)
	h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,3,3,1], strides=[1,2,2,1], padding = 'VALID')

	# fourth convolution layer
	W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 192, 192], stddev=0.1))
	b_conv4 = tf.Variable(tf.constant(0.1, shape = [192]))

	# hidden layer with ReLU
	h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv4, [1, 1, 1, 1], padding = 'SAME') + b_conv4)

	# fifth convolution layer
	W_conv5 = tf.Variable(tf.truncated_normal([3, 3, 192, 128], stddev=0.1))
	b_conv5 = tf.Variable(tf.constant(0.1, shape = [128]))

	# hidden layer with ReLU and pooling
	h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, [1, 1, 1, 1], padding = 'SAME') + b_conv5)
	h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding = 'VALID')

	#first fully-connected layer
	W_fcl1  = tf.Variable(tf.truncated_normal([6*6*128, 2048], stddev=0.1))
	b_fcl1 = tf.Variable(tf.constant(0.1, shape = [2048]))


	flatten_result = tf.reshape(h_pool5, [-1, 6*6*128])
	h_fcl1 = tf.nn.relu(tf.matmul(flatten_result, W_fcl1) + b_fcl1)

	#dropout
	keep_prob1 = 0.5
	fcl_dropout1 = tf.nn.dropout(h_fcl1, keep_prob1)

	#second fully-connected layer
	W_fcl2  = tf.Variable(tf.truncated_normal([2048, 2048], stddev=0.1))
	b_fcl2 = tf.Variable(tf.constant(0.1, shape = [2048]))

	h_fcl2 = tf.nn.relu(tf.matmul(h_fcl1, W_fcl2) + b_fcl2)

	#dropout
	keep_prob2 = 0.5
	fcl_dropout2 = tf.nn.dropout(h_fcl2, keep_prob2)

	#readout
	W_read  = tf.Variable(tf.truncated_normal([2048, 1000], stddev=0.1))
	b_read = tf.Variable(tf.constant(0.1, shape = [1000]))

	y_conv = tf.nn.softmax(tf.matmul(fcl_dropout2, W_read) + b_read)

	targets = tf.one_hot(targets, 1000)
	#TRAINING
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(y_conv)*targets, reduction_indices=[1]))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.argmax(targets,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	batchsize = 128
	decay = 0.0005
	momentum = 0.9
	learningrate = 0.1

	tf.initialize_all_variables().run(session=sess)
	print('reacjed')
	for i in range(10000):
		train_accuracy = accuracy.eval()
		print("accuracy %g" % (train_accuracy))
		train_step.run()

