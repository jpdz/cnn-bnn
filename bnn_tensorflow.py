from utils.inputdata import *
from utils.Gates import Gates
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
import pickle
hidden_layers = 3

import pickle as pc
# extract data
# inputsize1 each class <= 5000 lastyear data
# inputsize2 each class <= 900 app's data
gates = Gates(force = True, image_size = 28, inputsize1 = 916, inputsize2=916, mask = False)
gates.test.stack()
gates.valid.stack()

#store the image and label sperately
test_labels=gates.test.Y   
test_images=gates.test.X
valid_labels=gates.valid.Y   
valid_images=gates.valid.X


# weights trained from Lasagne
pickle_file_train = "mnist_bnn_paras.pkl"

with open(pickle_file_train, 'rb') as f:
	v=pickle.load(f)

x  = tf.placeholder(tf.float32, shape=[None, 28, 28], name = "input")
y_ = tf.placeholder(tf.float32, [None, 7])

def binarize(W):
    W = np.clip((W+1.)/2.,0,1)
    W = np.round(W)
    W = W * 2 - 1
    return W.astype(np.float32)

# Deterministic BinaryConnect (round to nearest)
def binaryDenselayer(inputs, w):
    inputs = tf.matmul(inputs, w)
    return inputs

def batchNormlayer(inputs, mean, invar, name="identity"): 
    inputs = (inputs-mean)/invar
    if name=="rectify":
        inputs = tf.maximum(inputs,0.0)    
    return inputs

# create model  
def model(inputs, weigths, means, invars):
    inputs = tf.reshape(inputs, [-1, 784])
    for i in range(hidden_layers):
    	inputs = binaryDenselayer(inputs, weigths[i])
        inputs = batchNormlayer(inputs,means[i], invars[i],"rectify")
    inputs = binaryDenselayer(inputs, weigths[3])
    inputs = batchNormlayer(inputs, means[3], invars[3])
    inputs = tf.nn.softmax(inputs, name="output")
    return inputs

weights = []
for w in v['weights']:
    weights.append(binarize(w))

v['weights'] = weights

# Construct model
y = model(x, v['weights'], v['means'], v['invars']) 

def generatePB(pb_dest = "model_mnist_bnn.pb"):
    gd = sess.graph.as_graph_def()
    gd2 = graph_util.convert_variables_to_constants(sess, gd, ['output'])
    with gfile.FastGFile(pb_dest, 'wb') as f:
        f.write(gd2.SerializeToString())
    print('pb saved')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: test_images,y_: test_labels}))
    print(sess.run(accuracy, feed_dict={x: valid_images,y_: valid_labels}))
    generatePB()






