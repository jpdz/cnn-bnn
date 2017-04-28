from utils.inputdata import *
from utils.Gates import Gates
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

model_name = "model_both.ckpt"


# extract data
# inputsize1 each class <= 5000
# inputsize2 each class <= 916
gates = Gates(force=True, image_size = 28, inputsize1=900, inputsize2=900, mask = False)

gates.train.stack()
gates.test.stack()
gates.valid.stack()

#print(gates.train.X[0])
#print(gates.test.X[0])
#print(gates.valid.X[0])

# input
x  = tf.placeholder(tf.float32, shape=[None, 28, 28], name = "input")
y_ = tf.placeholder(tf.float32, shape=[None, 7])

# create model
def multilayer(x, weights, biases):
	x_image = tf.reshape(x, [-1, 28, 28, 1])
	layer_conv_1 = tf.nn.conv2d(x_image, weights['conv1'], \
		strides=[1, 1, 1, 1], padding='VALID') + biases['conv1']
	layer_pool_1 = tf.nn.max_pool(layer_conv_1, ksize=[1, 2, 2, 1],\
		strides=[1, 2, 2, 1], padding='SAME')
	layer_conv_2 = tf.nn.conv2d(layer_pool_1, weights['conv2'], \
		strides=[1, 1, 1, 1], padding='VALID') + biases['conv2']
	layer_pool_2 = tf.nn.max_pool(layer_conv_2, ksize=[1, 2, 2, 1],\
		strides=[1, 2, 2, 1], padding='SAME')
	layer_fc_1 = tf.nn.relu(tf.matmul(tf.reshape(layer_pool_2, [-1, 4 * 4 * 50]),\
		weights['fc1']) + biases['fc1'])
	layer_fc_2 =tf.add(tf.matmul(layer_fc_1, weights['fc2']), biases['fc2'],name="output")

	return layer_fc_2

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)
def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

#store layers weight & bias
weights = {
	'conv1' : weight_variable([5,5,1,20], 'conv1_w'),
	'conv2' : weight_variable([5,5,20,50],'conv2_w'),
	'fc1'   : weight_variable([4 * 4 * 50, 400],'fc1_w'),
	'fc2'   : weight_variable([400,7], 'fc2_w')
} 

biases =  {
	'conv1' : bias_variable([20], 'conv1_b'),
	'conv2' : bias_variable([50], 'conv2_b'),
	'fc1'   : bias_variable([400],'fc1_b'),
	'fc2'   : bias_variable([7],  'fc2_b')	
}

# Construct model
y = multilayer(x,weights,biases)

# Define loss and Optimizer
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define initializer
init = tf.initialize_all_variables()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

def train(initial=True, model_src=model_name, 
	iteration = 20, batch_size=100, model_dest = model_name):
	if initial:
		sess.run(init)
	else:
		saver.restore(sess, model_src)

	for j in range(iteration):
		for i in range(batch_size, gates.train.num, batch_size):
			if i%1000 == 0:
				train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_: batch_y})
				print("iteration %d step %d, training accuracy %g"%(j, i, train_accuracy))
			batch_x, batch_y = gates.train.batch(batch_size)
			sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
		save_path = saver.save(sess, model_dest)
		print("Model saved in file: %s" % save_path)
		print("Orignial test accuray is %g"%accuracy.eval({x: gates.test.X, y_:gates.test.Y}))
		print("New test accuray is %g"%accuracy.eval({x: gates.valid.X, y_:gates.valid.Y}))
		test()

def test(train=False, model_src=model_name):
	label = ["AND", "NAND", "NOR", "NOT", "OR", "XNOR", "XOR"]
	if not train:
		saver.restore(sess, model_src)
	l1 = len(gates.valid.X)
	l2 = len(gates.test.X)
	l1_seg = l1/7
	l2_seg = l2/7
	print("##################### New Test Set ##########################")
	for k,i in enumerate(range(0,l1,l1_seg)):
		print("test accuracy for %4s is %g"%(label[k],accuracy.eval({x: gates.valid.X[i:i+l1_seg], y_:gates.valid.Y[i:i+l1_seg]})))
	print("##################### Old Test Set ##########################")
	for k,i in enumerate(range(0,9996,1428)):
		print("test accuracy for %4s is %g"%(label[k],accuracy.eval({x: gates.test.X[i:i+1428], y_:gates.test.Y[i:i+1428]})))
	#print(accuracy.eval({x: gates.train.X, y_:gates.train.Y}))

def generatePB(pb_dest = "model.pb"):
	gd = sess.graph.as_graph_def()
	gd2 = graph_util.convert_variables_to_constants(sess, gd, ['output'])
	with gfile.FastGFile(pb_dest, 'wb') as f:
		f.write(gd2.SerializeToString())
	print('pb saved')

def test2(model_src = model_name):	
	saver.restore(sess, model_src)
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct = correct_prediction.eval({x: gates.valid.X, y_:gates.valid.Y})
	print(correct)
	y_num = y.eval({x:gates.valid.X})
	print(y_num)
	for i,res in enumerate(correct):
		if not res:
			print "actual res is %d, prediction res is %d"%(np.argmax(gates.test.Y[i]), np.argmax(y_num[i]))
 
	

# Running
with tf.Session() as sess:
	#train(initial=True, iteration=45)
	#test2()
	test()
	generatePB()


