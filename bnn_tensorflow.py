from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import numpy as np
from Gates import Gates
from inputdata import *

hidden_layers = 3
# extract data
# inputsize1 each class <= 5000 lastyear data
# inputsize2 each class <= 916 app's data
gates = Gates(force = True, image_size = 28, inputsize1 = 916, inputsize2=916, mask = False)
gates.test.stack()
gates.valid.stack()

#store the image and label sperately
test_labels=gates.test.Y   
test_images=gates.test.X
valid_labels=gates.valid.Y   
valid_images=gates.valid.X
test_images  = test_images.reshape(-1, 784)
valid_images  = valid_images.reshape(-1, 784)

# weights trained from Lasagne
pickle_file_train = "weights3.pickle"

with open(pickle_file_train, 'rb') as f:
	weigths=pickle.load(f)

x = tf.placeholder(tf.float32, [None, 784], name = "inputs")
y_ = tf.placeholder(tf.float32, [None, 7])

# Deterministic BinaryConnect (round to nearest)
def binarize(W):
    W = np.clip((W+1.)/2.,0,1)
    W = np.round(W)
    W = W * 2 - 1
    return W

def binaryDenselayer(inputs, w, b):
    sess = tf.Session()
    w=binarize(w)
    w = tf.constant(w, tf.float32)
    inputs = tf.matmul(inputs, w)+b
    return inputs

def batchNormlayer(inputs, size, name="identity"):    
    mean, var = tf.nn.moments(inputs, axes=[0])
    outputsize = inputs.get_shape().as_list()[0]
    scale = tf.Variable(tf.ones([size]))
    shift = tf.Variable(tf.zeros([size]))
    epsilon = 1e-3
    inputs = tf.nn.batch_normalization(inputs, mean, var, shift, scale, epsilon)
    if name=="rectify":
        inputs = tf.maximum(inputs,0.0)
        print "rectify"
    return inputs

# create model  
def model(inputs, weigths):
    for i in range(hidden_layers):
    	inputs = binaryDenselayer(inputs, weigths[i], weigths[i+4])
        inputs = batchNormlayer(inputs,weigths[i].shape[1],"rectify")
    inputs = binaryDenselayer(inputs, weigths[3],weigths[7])
    inputs = batchNormlayer(inputs,weigths[3].shape[1])
    return inputs

# Construct model
y = model(x, weigths) 

with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(correct_prediction, feed_dict={x: test_images,y_: test_labels}))
	print(sess.run(accuracy, feed_dict={x: test_images,y_: test_labels}))
        print(sess.run(accuracy, feed_dict={x: valid_images,y_: valid_labels}))
	#print(sess.run(accuracy, feed_dict={x: valid_images[800:],y_: valid_labels[800:]}))
        '''
        for i in range(0,1596,228):
		print(sess.run(accuracy, feed_dict={x: valid_images[i:i+228],y_: valid_labels[i:i+228]}))
	for i in range(0,9996,1428):
		print(sess.run(accuracy, feed_dict={x: test_images[i:i+1428],y_: test_labels[i:i+1428]}))

	'''




