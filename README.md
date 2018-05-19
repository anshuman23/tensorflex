# Tensorflex

## Contents
- [__How to run__](https://github.com/anshuman23/tensorflex/#how-to-run)
- [__Documentation__](https://github.com/anshuman23/tensorflex/#documentation) 
- [__Pull Requests Made__](https://github.com/anshuman23/tensorflex/#pull-requests-made) 



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
    iex(1)> graph = Tensorflex.read_graph("classify_image_graph_def.pb")
    2018-05-17 23:36:16.488469: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that    this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA 2018-05-17 23:36:16.774442: W             tensorflow/core/framework/op_def_util.cc:334] OpBatchNormWithGlobalNormalization is deprecated. It will cease to work in   GraphDef version 9. Use tf.nn.batch_normalization().
    Successfully imported graph
    #Reference<0.1610607974.1988231169.250293>
    ```
    
### Pull Requests Made 
- In chronological order:
    - [PR #2: Renamed app to Tensorflex from TensorflEx](https://github.com/anshuman23/tensorflex/pull/2) 
    - [PR #3: Added support for reading pretrained graph definition files](https://github.com/anshuman23/tensorflex/pull/3)
    - [PR #4: Merged all support functions into read_graph; returning atoms](https://github.com/anshuman23/tensorflex/pull/4)
    - [PR #6: Returning list of op names in get_graph_ops and extended error atoms to all TF error codes](https://github.com/anshuman23/tensorflex/pull/6)
