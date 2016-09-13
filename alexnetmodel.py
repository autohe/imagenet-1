import tensorflow as tf
import numpy as np



num_targets = 1000
random = np.random.randn(1, 224,224,3)
random_target = np.random.randn(1, num_targets)

class AlexNet:
	def __init__(self):
		self.keep_prob1 = tf.placeholder(tf.float32, [], name='keep_prob1')
		self.keep_prob2 = tf.placeholder(tf.float32, [], name='keep_prob2')

	def build(self, input_data, targets, eval=True):
		with tf.variable_scope('AlexNet') as scope:
			if eval:
				scope.reuse_variables()
			# first convolution layer
			W_conv1 = tf.get_variable('w_conv1', shape=[11, 11, 3, 48], initializer=tf.truncated_normal_initializer(stddev=0.1))
			b_conv1 = tf.get_variable('b_conv1', shape=[48], initializer=tf.constant_initializer(0.1))

			x_image = tf.reshape(input_data, [-1, 224, 224, 3])

			# hidden layer with ReLU and max pooling
			h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, [1, 4, 4, 1], padding = 'SAME') + b_conv1)

			# second convolution layer
			W_conv2 = tf.get_variable('w_conv2', shape=[5, 5, 48, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
			b_conv2 = tf.get_variable('b_conv2', shape=[128], initializer=tf.constant_initializer(0.1))

			# hidden layer with ReLU and max pooling
			h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, [1, 1, 1, 1], padding = 'SAME') + b_conv2)
			h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1,3,3,1], strides=[1,2,2,1], padding = 'VALID')

			# third convolution layer
			W_conv3 = tf.get_variable('w_conv3', shape=[3, 3, 128, 192], initializer=tf.truncated_normal_initializer(stddev=0.1))
			b_conv3 = tf.get_variable('b_conv3', shape=[192], initializer=tf.constant_initializer(0.1))

			# hidden layer with ReLU
			h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, [1, 1, 1, 1], padding = 'SAME') + b_conv3)
			h_pool3 = tf.nn.max_pool(h_conv3, ksize=[1,3,3,1], strides=[1,2,2,1], padding = 'VALID')

			# fourth convolution layer
			W_conv4 = tf.get_variable('w_conv4', shape=[3, 3, 192, 192], initializer=tf.truncated_normal_initializer(stddev=0.1))
			b_conv4 = tf.get_variable('b_conv4', shape=[192], initializer=tf.constant_initializer(0.1))

			# hidden layer with ReLU
			h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv4, [1, 1, 1, 1], padding = 'SAME') + b_conv4)

			# fifth convolution layer
			W_conv5 = tf.get_variable('w_conv5', shape=[3, 3, 192, 128], initializer=tf.truncated_normal_initializer(stddev=0.1))
			b_conv5 = tf.get_variable('b_conv5', shape=[128], initializer=tf.constant_initializer(0.1))

			# hidden layer with ReLU and pooling
			h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, [1, 1, 1, 1], padding = 'SAME') + b_conv5)
			h_pool5 = tf.nn.max_pool(h_conv5, ksize=[1,3,3,1], strides=[1,2,2,1], padding = 'VALID')

			#first fully-connected layer

			W_fcl1 = tf.get_variable('w_fcl1', shape=[6*6*128, 2048], initializer=tf.truncated_normal_initializer(stddev=0.1))
			b_fcl1 = tf.get_variable('b_fcl1', shape=[2048], initializer=tf.constant_initializer(0.1))


			flatten_result = tf.reshape(h_pool5, [-1, 6*6*128])
			h_fcl1 = tf.nn.relu(tf.matmul(flatten_result, W_fcl1) + b_fcl1)

			#dropout
			fcl_dropout1 = tf.nn.dropout(h_fcl1, self.keep_prob1)

			#second fully-connected layer

			W_fcl2 = tf.get_variable('w_fcl2', shape=[2048, 2048], initializer=tf.truncated_normal_initializer(stddev=0.1))
			b_fcl2 = tf.get_variable('b_fcl2', shape=[2048], initializer=tf.constant_initializer(0.1))

			h_fcl2 = tf.nn.relu(tf.matmul(h_fcl1, W_fcl2) + b_fcl2)

			#dropout
			fcl_dropout2 = tf.nn.dropout(h_fcl2, self.keep_prob2)
		
			#readout
			W_read = tf.get_variable('w_read', shape=[2048, 1000], initializer=tf.truncated_normal_initializer(stddev=0.1))
			b_read = tf.get_variable('b_read', shape=[1000], initializer=tf.constant_initializer(0.1))

			y_conv = tf.nn.softmax(tf.matmul(fcl_dropout2, W_read) + b_read)

			targets = tf.one_hot(targets, 1000)
			
			if not eval:
				#TRAINING
				cross_entropy = tf.reduce_mean(-tf.reduce_sum(tf.log(y_conv + 1e-16)*targets, reduction_indices=[1]))
				train_step = tf.train.AdamOptimizer(learning_rate = 0.1, beta1 = 0.0005).minimize(cross_entropy)
				return cross_entropy, train_step

			correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.argmax(targets,1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



			return accuracy


	

