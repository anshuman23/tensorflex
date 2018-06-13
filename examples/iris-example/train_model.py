import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.tools import freeze_graph
import os

dir = os.path.dirname(os.path.realpath(__file__)) + '/data/'

seed = 1234
np.random.seed(seed)
tf.set_random_seed(seed)

dataset = pd.read_csv('dataset.csv')
dataset = pd.get_dummies(dataset, columns=['Species']) 
values = list(dataset.columns.values)

y = dataset[values[-3:]]
y = np.array(y, dtype='float32')
X = dataset[values[1:-3]]
X = np.array(X, dtype='float32')

indices = np.random.choice(len(X), len(X), replace=False)
X_values = X[indices]
y_values = y[indices]

test_size = 10
X_test = X_values[-test_size:]
X_train = X_values[:-test_size]
y_test = y_values[-test_size:]
y_train = y_values[:-test_size]

sess = tf.Session()

interval = 50
epoch = 500

X_data = tf.placeholder(shape=[None, 4], dtype=tf.float32, name='input')
y_target = tf.placeholder(shape=[None, 3], dtype=tf.float32)

hidden_layer_nodes = 8

w1 = tf.Variable(tf.random_normal(shape=[4,hidden_layer_nodes]), name='weights1') 
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]), name='biases1')   
w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,3]), name='weights2') 
b2 = tf.Variable(tf.random_normal(shape=[3]),name='biases2')


hidden_output = tf.nn.relu(tf.add(tf.matmul(X_data, w1), b1))
final_output = tf.nn.softmax(tf.add(tf.matmul(hidden_output, w2), b2), name='output')


loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(final_output), axis=0))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)


init = tf.global_variables_initializer()
sess.run(init)

saver = tf.train.Saver()

print('Training the model...')
for i in range(1, (epoch + 1)):
    sess.run(optimizer, feed_dict={X_data: X_train, y_target: y_train})
    if i % interval == 0:
        print('Epoch', i, '|', 'Loss:', sess.run(loss, feed_dict={X_data: X_train, y_target: y_train}))

tf.train.write_graph(sess.graph_def, dir, 'graph.pbtxt')
saver.save(sess, dir+'graph')
freeze_graph.freeze_graph(dir+'graph.pbtxt','', False, dir+'graph', 'output', 'save/restore_all', 'save/Const:0', 'data/graphdef.pb', True, '')

for i in range(len(X_test)):
    print('For datapoint: '+ str(X_test[i])+ ' Actual:', y_test[i], 'Predicted:', np.rint(sess.run(final_output, feed_dict={X_data: [X_test[i]]})))

