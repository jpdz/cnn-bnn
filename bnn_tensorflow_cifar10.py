from utils.Gates import Gates
from utils.inputdata import *
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

gates = Gates(force = True, image_size = 28, inputsize1 = 916, inputsize2=916, mask = False)

gates.test.stack()
gates.valid.stack()
gates.test.resize2()
gates.valid.resize2()


x  = tf.placeholder(tf.float32, shape=[None, 32, 32], name = "input")
y_ = tf.placeholder(tf.float32, [None, 7])

print("Loading the trained parameters and binarizing the weights...")
num_layers = 9

with np.load( 'model_all.npz') as f:
	h = f['arr_%d' % 0] 

weights = []
bias = []
mean = []
std = []
beta = []
gamma = []


for i in range(0,num_layers * 6,6):

	weights.append(h[i].astype(np.float32))
	bias.append(h[i+1].astype(np.float32))
	t = h[i+2].shape[1]
	mean.append(h[i+2].reshape(t))
	std.append(h[i+3].reshape(t))
	beta.append(h[i+4].reshape(t))
	gamma.append(h[i+5].reshape(t))



def binarize(W):
    W = np.clip((W+1.)/2.,0,1)
    W = np.round(W)
    W = W * 2 - 1
    return W.astype(np.float32)

for i,w in enumerate(weights):
	if i<6:
		weights[i] = np.flip(np.flip(w, axis=2), axis=3)
		weights[i] = np.transpose(weights[i],[2,3,1,0])

Bweights = [binarize(w) for w in weights]


def batchNormlayer(inputs, mean, std, beta, gamma, name="rectify"):    
    inputs =  (inputs - mean) * (gamma / std) + beta
    if name=="rectify":
        inputs = tf.maximum(inputs,0.0)
    return inputs


def cnn(x, weights, means, stds, betas, gammas):
	x = tf.reshape(x, [-1, 32, 32, 1])

	x = tf.nn.conv2d(x, weights[0], [1,1,1,1], "SAME")
	x = batchNormlayer(x, means[0], stds[0], betas[0], gammas[0])
	x = tf.nn.conv2d(x, weights[1], [1,1,1,1], "SAME")
	x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],\
		strides=[1, 2, 2, 1], padding='SAME')
	x = batchNormlayer(x, means[1], stds[1], betas[1], gammas[1])

	x = tf.nn.conv2d(x, weights[2], [1,1,1,1], "SAME")
	x = batchNormlayer(x, means[2], stds[2], betas[2], gammas[2])
	x = tf.nn.conv2d(x, weights[3], [1,1,1,1], "SAME")
	x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],\
		strides=[1, 2, 2, 1], padding='SAME')
	x = batchNormlayer(x, means[3], stds[3], betas[3], gammas[3])

	
	x = tf.nn.conv2d(x, weights[4], [1,1,1,1], "SAME")
	x = batchNormlayer(x, means[4], stds[4], betas[4], gammas[4])
	x = tf.nn.conv2d(x, weights[5], [1,1,1,1], "SAME")

	x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],\
		strides=[1, 2, 2, 1], padding='SAME')
	x = batchNormlayer(x, means[5], stds[5], betas[5], gammas[5])

	x = tf.transpose(x, [0, 3, 1, 2])
	x = tf.reshape(x, [-1, 4*4*512])
	
	x = tf.matmul(x, weights[6])
	x = batchNormlayer(x, means[6], stds[6], betas[6], gammas[6])

	x = tf.matmul(x, weights[7])
	x = batchNormlayer(x, means[7], stds[7], betas[7], gammas[7])

	x = tf.matmul(x, weights[8])
	x = batchNormlayer(x, means[8], stds[8], betas[8], gammas[8], name="identity")
	
	x = tf.nn.softmax(x, name="output")
	
	return x

y = cnn(x, Bweights, mean, std, beta, gamma)

def generatePB(pb_dest = "cifar_bnn_new.pb"):
    gd = sess.graph.as_graph_def()
    gd2 = graph_util.convert_variables_to_constants(sess, gd, ['output'])
    with gfile.FastGFile(pb_dest, 'wb') as f:
        f.write(gd2.SerializeToString())
    print('pb saved')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: gates.valid.X[0:1000], y_: gates.valid.Y[0:1000]}))
    generatePB()




