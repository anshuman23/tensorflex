import tensorflow as tf
import os

model_dir = './model/'
output_node_names = 'add'

checkpoint = tf.train.get_checkpoint_state(model_dir)
input_checkpoint = checkpoint.model_checkpoint_path

absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
output_graph = absolute_model_dir + "/frozen_model_lstm.pb"

clear_devices = True

with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
    saver.restore(sess, input_checkpoint)
    output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), output_node_names.split(",")) 
        
    with tf.gfile.GFile(output_graph, "wb") as f:
         f.write(output_graph_def.SerializeToString())
     
    print("%d ops in the final graph." % len(output_graph_def.node))
                
