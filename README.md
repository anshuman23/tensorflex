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

- __Reading in a pretrained graph defintion file__

    This is the first step of the Inference process in Tensorflow/Tensorflex. For this example we read in the Inception Google model available for download [here](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz). The file name is `classify_image_graph_def.pb`. The example is as follows:
    ```elixir
    iex(1)> {:ok, graph} = Tensorflex.read_graph("classify_image_graph_def.pb")
    2018-05-17 23:36:16.488469: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that    this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA 2018-05-17 23:36:16.774442: W             tensorflow/core/framework/op_def_util.cc:334] OpBatchNormWithGlobalNormalization is deprecated. It will cease to work in   GraphDef version 9. Use tf.nn.batch_normalization().
    Successfully imported graph
    #Reference<0.1610607974.1988231169.250293>
    
    iex(2)> op_list = Tensorflex.get_graph_ops graph
    ["softmax/biases", "softmax/weights", "pool_3/_reshape/shape",
    "mixed_10/join/concat_dim", "mixed_10/tower_2/conv/batchnorm/moving_variance",
    "mixed_10/tower_2/conv/batchnorm/moving_mean",
    "mixed_10/tower_2/conv/batchnorm/gamma",
    "mixed_10/tower_2/conv/batchnorm/beta", "mixed_10/tower_2/conv/conv2d_params",
    "mixed_10/tower_1/mixed/conv_1/batchnorm/moving_variance",
    "mixed_10/tower_1/mixed/conv_1/batchnorm/moving_mean",
    "mixed_10/tower_1/mixed/conv_1/batchnorm/gamma",
    "mixed_10/tower_1/mixed/conv_1/batchnorm/beta",
    "mixed_10/tower_1/mixed/conv_1/conv2d_params",
    "mixed_10/tower_1/mixed/conv/batchnorm/moving_variance",
    "mixed_10/tower_1/mixed/conv/batchnorm/moving_mean",
    "mixed_10/tower_1/mixed/conv/batchnorm/gamma",
    "mixed_10/tower_1/mixed/conv/batchnorm/beta",
    "mixed_10/tower_1/mixed/conv/conv2d_params",
    "mixed_10/tower_1/conv_1/batchnorm/moving_variance",
    "mixed_10/tower_1/conv_1/batchnorm/moving_mean",
    "mixed_10/tower_1/conv_1/batchnorm/gamma",
    "mixed_10/tower_1/conv_1/batchnorm/beta",
    "mixed_10/tower_1/conv_1/conv2d_params",
    "mixed_10/tower_1/conv/batchnorm/moving_variance",
    "mixed_10/tower_1/conv/batchnorm/moving_mean",
    "mixed_10/tower_1/conv/batchnorm/gamma",
    "mixed_10/tower_1/conv/batchnorm/beta", "mixed_10/tower_1/conv/conv2d_params",
    "mixed_10/tower/mixed/conv_1/batchnorm/moving_variance",
    "mixed_10/tower/mixed/conv_1/batchnorm/moving_mean",
    "mixed_10/tower/mixed/conv_1/batchnorm/gamma",
    "mixed_10/tower/mixed/conv_1/batchnorm/beta",
    "mixed_10/tower/mixed/conv_1/conv2d_params",
    "mixed_10/tower/mixed/conv/batchnorm/moving_variance",
    "mixed_10/tower/mixed/conv/batchnorm/moving_mean",
    "mixed_10/tower/mixed/conv/batchnorm/gamma",
    "mixed_10/tower/mixed/conv/batchnorm/beta",
    "mixed_10/tower/mixed/conv/conv2d_params",
    "mixed_10/tower/conv/batchnorm/moving_variance",
    "mixed_10/tower/conv/batchnorm/moving_mean",
    "mixed_10/tower/conv/batchnorm/gamma", "mixed_10/tower/conv/batchnorm/beta",
    "mixed_10/tower/conv/conv2d_params", "mixed_10/conv/batchnorm/moving_variance",
    "mixed_10/conv/batchnorm/moving_mean", "mixed_10/conv/batchnorm/gamma",
    "mixed_10/conv/batchnorm/beta", "mixed_10/conv/conv2d_params",
    "mixed_9/join/concat_dim", ...]
    ```
    
    
- __Matrix capabilities__
    - Matrices are created using `create_matrix` which takes `number of rows`, `number of columns` and list(s) of matrix data as inputs
    - `matrix_pos` help get the value stored in the matrix at a particular row and column
    - `size_of_matrix` returns a tuple of the size of matrix as `{number of rows, number of columns}`
    - `matrix_to_lists` returns the data of the matrix as list of lists
    ```elixir
    iex(1)> m = Tensorflex.create_matrix(2,3,[[2.2,1.3,44.5],[5.5,6.1,3.333]])
    #Reference<0.1012898165.3475636225.187946>

    iex(2)> Tensorflex.matrix_pos(m,2,1)
    5.5

    iex(3)> Tensorflex.size_of_matrix m
    {2, 3}

    iex(4)> Tensorflex.matrix_to_lists m
    [[2.2, 1.3, 44.5], [5.5, 6.1, 3.333]]
    
 - __Tensor usage__
    - Numeral Tensors:
        - Basically `float_tensor` handles numeral tensors. It has two variants: one that takes in just 1 argument and the other which takes in 2 arguments
        - The one which takes 1 argument is just for making a tensor out of a single number
        - The 2 argument variant is actually more important and is used for multidimensional Tensors
        - Here, the first argument is the values and the second consists of the dimensions of the Tensor. Both these are matrices

    ```elixir
    iex(1)> dims = Tensorflex.create_matrix(1,3,[[1,1,3]])
    #Reference<0.3771206257.3662544900.104749>

    iex(2)> vals = Tensorflex.create_matrix(1,3,[[245,202,9]])
    #Reference<0.3771206257.3662544900.104769>

    iex(3)> Tensorflex.float_tensor 123.12
    {:ok, #Reference<0.3771206257.3662544897.110716>}

    iex(4)> {:ok, ftensor} = Tensorflex.float_tensor(vals,dims)
    {:ok, #Reference<0.3771206257.3662544897.111510>}

    iex(5)> Tensorflex.tensor_datatype ftensor
    {:ok, :tf_double}
    ```
    
    - String Tensors:
    
    ```elixir
    iex(1)> {:ok, str_tensor} = Tensorflex.string_tensor "1234"
    {:ok, #Reference<0.1771877210.87949316.135871>}
    
    iex(2)> Tensorflex.tensor_datatype str_tensor
    {:ok, :tf_string}


    ```
    
    
### Pull Requests Made 
- In chronological order:
    - [PR #2: Renamed app to Tensorflex from TensorflEx](https://github.com/anshuman23/tensorflex/pull/2) 
    - [PR #3: Added support for reading pretrained graph definition files](https://github.com/anshuman23/tensorflex/pull/3)
    - [PR #4: Merged all support functions into read_graph; returning atoms](https://github.com/anshuman23/tensorflex/pull/4)
    - [PR #6: Returning list of op names in get_graph_ops and extended error atoms to all TF error codes](https://github.com/anshuman23/tensorflex/pull/6)
    - [PR #7: Added tensor support for strings and for getting TF_DataType](https://github.com/anshuman23/tensorflex/pull/7)
    - [PR #8: Added matrix functions, numeral tensors, better returns & removed unnecessary POC code](https://github.com/anshuman23/tensorflex/pull/8)
    - [PR #9: Added freeze graph Python example](https://github.com/anshuman23/tensorflex/pull/9)
