# TensorflEx

## Contents
- [__How to run__](https://github.com/anshuman23/tensorflex/#how-to-run)
- [__Documentation__](https://github.com/anshuman23/tensorflex/#documentation) 


### How to run
- You need to have the Tensorflow C API installed. Look [here](https://www.tensorflow.org/install/install_c) for details.
- Clone this repository and `cd` into it
- Run `mix deps.get` to install the dependencies
- Run `mix compile` to compile the code
- Open up `iex` using `iex -S mix`


### Documentation

- __Example 1: Reading in a pretrained graph defintion file__

    This is the first step of the Inference process in Tensorflow/Tensorflex. For this example we read in the Inception Google model available for download [here](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz). The file name is `classify_image_graph_def.pb`. The example is as follows:
    ```elixir
    iex(1)> graph = Tensorflex.new_graph
    2018-05-16 00:46:18.446563: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
    #Reference<0.147544838.3696885761.125951>

    iex(2)> graph_opts = Tensorflex.new_import_graph_def_opts
    #Reference<0.147544838.3696885761.126031>
    
    iex(3)> graph_def = Tensorflex.read_file("classify_image_graph_def.pb")
    #Reference<0.147544838.3696885761.126074>
    
    iex(4)> Tensorflex.graph_import_graph_def(graph, graph_def, graph_opts)
    2018-05-16 00:50:34.497120: W tensorflow/core/framework/op_def_util.cc:334] Op BatchNormWithGlobalNormalization is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
    Successfully imported graph
    '[INFO] Graph loaded\n'
    ```
