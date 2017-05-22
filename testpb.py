from utils.inputdata import *
from utils.Gates import Gates


pbfile = "model.pb"

# extract data
gates = Gates(force=True, imagesize = 28)

gates.train.stack()
gates.test.stack()

y_ = tf.placeholder(tf.float32, shape=[None, 7])


with tf.Session() as sess:
	with gfile.FastGFile(pbfile,'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name="")

	# get input 
	input_data  = sess.graph.get_tensor_by_name("input:0")
	# get output
	output_data = sess.graph.get_tensor_by_name("output:0")
	# feed input, run test
	y = sess.run(output_data, {input_data: gate.test.X})
	# test result
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(accuracy.eval({y_:gates.test.Y}))

