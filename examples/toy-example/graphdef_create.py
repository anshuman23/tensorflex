import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import os

dir = os.path.dirname(os.path.realpath(__file__)) + '/data/'

input = tf.placeholder(tf.float32, shape=[1,3], name='input') 
weights = tf.Variable(tf.zeros_initializer(shape=[3,2]), dtype=tf.float32, name='weights')
biases = tf.Variable(tf.zeros_initializer(shape=[2]), dtype=tf.float32, name='biases')
output = tf.nn.relu(tf.matmul(input, weights) + biases, name='output')
saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)
  tf.train.write_graph(sess.graph_def, dir, 'graph.pbtxt')
  sess.run(tf.assign(weights, [[1.43, 22.1],[44.8,5.76],[7.12,8.4]]))
  sess.run(tf.assign(biases, [3,3]))
  saver.save(sess, dir + 'graph')

freeze_graph.freeze_graph(dir+'graph.pbtxt','', False, dir+'graph', 'output', 'save/restore_all', 'save/Const:0', 'data/graphdef.pb', True, '')
