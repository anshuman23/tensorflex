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

### Example
Examples are generally added in full description on my blog [here](http://anshumanc.ml). A blog post covering how to do classification on the Iris Dataset is present [here](http://www.anshumanc.ml/gsoc/2018/06/14/gsoc/).

Here we will briefly touch upon how to use the Google V3 Inception pre-trained graph model to do image classficiation from over a 1000 classes. First, the Inception V3 model can be downloaded here: http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

After unzipping, see that it contains the graphdef .pb file (`classify_image_graphdef.pb`) which contains our graph definition, a test jpeg image that should identify/classify as a panda (`cropped_panda.pb`) and a few other files I will detail later.

Now for running this in Tensorflex first the graph is loaded:

```elixir
iex(1)> {:ok, graph} = Tensorflex.read_graph("classify_image_graph_def.pb")
2018-07-02 00:27:40.532461: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA 2018-07-02 00:27:40.641913: W tensorflow/core/framework/op_def_util.cc:334] OpBatchNormWithGlobalNormalization is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
{:ok, #Reference<0.101807160.3449159683.146711>}
```
Then the cropped_panda image is loaded using the new `load_image_as_tensor` function:
```elixir
iex(2)> {:ok, input_tensor} = Tensorflex.load_image_as_tensor("cropped_panda.jpg")
{:ok, #Reference<0.101807160.3449159683.146733>}
```
Then create the output tensor which will hold out output vector values. For the inception model, the output is received as a 1008x1 tensor, as there are 1008 classes in the model:
```elixir
iex(3)> out_dims = Tensorflex.create_matrix(1,2,[[1008,1]])
#Reference<0.101807160.3449159683.146741>

iex(4)> {:ok, output_tensor} = Tensorflex.float32_tensor_alloc(out_dims)
{:ok, #Reference<0.101807160.3449159683.146748>}
```
Then the output results are read into a list called `results`. Also, the input operation in the Inception model is `DecodeJpeg` and the output operation is `softmax`:
```elixir
iex(5)> results = Tensorflex.run_session(graph, input_tensor, output_tensor, "DecodeJpeg", "softmax")
[
  [1.059142014128156e-4, 2.8240500250831246e-4, 8.30648496048525e-5,
   1.2982363114133477e-4, 7.32232874725014e-5, 8.014426566660404e-5,
   6.63459359202534e-5, 0.003170756157487631, 7.931600703159347e-5,
   3.707312498590909e-5, 3.0997329304227605e-5, 1.4232713147066534e-4,
   1.0381334868725389e-4, 1.1057958181481808e-4, 1.4321311027742922e-4,
   1.203602587338537e-4, 1.3130248407833278e-4, 5.850398520124145e-5,
   2.641105093061924e-4, 3.1629020668333396e-5, 3.906813799403608e-5,
   2.8646905775531195e-5, 2.2863158665131778e-4, 1.2222197256051004e-4,
   5.956588938715868e-5, 5.421260357252322e-5, 5.996063555357978e-5,
   4.867801326327026e-4, 1.1005574924638495e-4, 2.3433618480339646e-4,
   1.3062104699201882e-4, 1.317620772169903e-4, 9.388553007738665e-5,
   7.076268957462162e-5, 4.281177825760096e-5, 1.6863139171618968e-4,
   9.093972039408982e-5, 2.611844101920724e-4, 2.7584232157096267e-4,
   5.157176201464608e-5, 2.144951868103817e-4, 1.3628098531626165e-4,
   8.007588621694595e-5, 1.7929042223840952e-4, 2.2831936075817794e-4,
   6.216531619429588e-5, 3.736453436431475e-5, 6.782123091397807e-5,
   1.1538144462974742e-4, ...]
]
``` 
Finally, we need to find which class has the maximum probability and identify it's label. Since results is a List of Lists, it's better to read in the nested list. Then we need to find the index of the element in the new list which as the maximum value. Therefore:
```elixir
iex(6)> actual_results = Enum.max(results)
  [1.059142014128156e-4, 2.8240500250831246e-4, 8.30648496048525e-5,
   1.2982363114133477e-4, 7.32232874725014e-5, 8.014426566660404e-5,
   6.63459359202534e-5, 0.003170756157487631, 7.931600703159347e-5,
   3.707312498590909e-5, 3.0997329304227605e-5, 1.4232713147066534e-4,
   1.0381334868725389e-4, 1.1057958181481808e-4, 1.4321311027742922e-4,
   1.203602587338537e-4, 1.3130248407833278e-4, 5.850398520124145e-5,
   2.641105093061924e-4, 3.1629020668333396e-5, 3.906813799403608e-5,
   2.8646905775531195e-5, 2.2863158665131778e-4, 1.2222197256051004e-4,
   5.956588938715868e-5, 5.421260357252322e-5, 5.996063555357978e-5,
   4.867801326327026e-4, 1.1005574924638495e-4, 2.3433618480339646e-4,
   1.3062104699201882e-4, 1.317620772169903e-4, 9.388553007738665e-5,
   7.076268957462162e-5, 4.281177825760096e-5, 1.6863139171618968e-4,
   9.093972039408982e-5, 2.611844101920724e-4, 2.7584232157096267e-4,
   5.157176201464608e-5, 2.144951868103817e-4, 1.3628098531626165e-4,
   8.007588621694595e-5, 1.7929042223840952e-4, 2.2831936075817794e-4,
   6.216531619429588e-5, 3.736453436431475e-5, 6.782123091397807e-5,
   1.1538144462974742e-4, ...]

iex(7)> maxval_class = Enum.max(results) |> Enum.max()
0.8849328756332397

iex(8)> Enum.find_index(actual_results, fn(x) -> x == maxval_class end)
169
``` 

We can thus see that the class with the maximum probability predicted (__0.8849328756332397__) for the image is __169__. We will now find what the 169 label corresponds to. For this we can look back into the unzipped Inception folder, where there is a file called `imagenet_2012_challenge_label_map_proto.pbtxt`. On opening this file, we can find the string class identifier for the `169` class index. This is `n02510455` and is present on Line 1556 in the file. Finally, we need to match this string identifier to a set of identification labels by referring to the file `imagenet_synset_to_human_label_map.txt` file. Here we can see that corresponding to the string class `n02510455` the human labels are `giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca` (Line 3691 in the file). 

Thus, we have correctly identified the animal in the image as a panda using Tensorflex! 

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
        - Basically `float64_tensor` handles numeral tensors. It has two variants: one that takes in just 1 argument and the other which takes in 2 arguments
        - The one which takes 1 argument is just for making a tensor out of a single number
        - The 2 argument variant is actually more important and is used for multidimensional Tensors
        - Here, the first argument is the values and the second consists of the dimensions of the Tensor. Both these are matrices

    ```elixir
    iex(1)> dims = Tensorflex.create_matrix(1,3,[[1,1,3]])
    #Reference<0.3771206257.3662544900.104749>

    iex(2)> vals = Tensorflex.create_matrix(1,3,[[245,202,9]])
    #Reference<0.3771206257.3662544900.104769>

    iex(3)> Tensorflex.float64_tensor 123.12
    {:ok, #Reference<0.3771206257.3662544897.110716>}

    iex(4)> {:ok, ftensor} = Tensorflex.float64_tensor(vals,dims)
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
 - __Running sessions__
    - Sessions are basically used to run a set of inputs all through the operations in a predefined graph and obtain prediction outputs
    - To exemplify the working of the entire prediction pipeline, I am going to use the simple toy graph created by the script in `examples/toy-example/` called `graphdef_create.py` . Upon running the script you should have a graph definition file called `graphdef.pb`. You can also download this file stored in my Dropbox [here](https://www.dropbox.com/s/r7n6duan70s7scb/graphdef.pb?dl=0).

    - Then I would recommend going through the `graphdef_create.py` file to get an idea of what the operations are. The code basically works like a very simple matrix multiplication of some predefined weights with the input and then the addition of the biases. Ideally the weights should be ascertained through training but since this is a toy example they are predefined [(look here in the code)](https://github.com/anshuman23/tensorflex/blob/master/examples/graphdef_create.py#L17-L18).

    - The more important thing to notice in the `graphdef_create.py` file are the operations where the input is fed and where the output is obtained. It is important to know the names of these operations as when we perform Inference the input will be fed to that named input operation and the output will be obtained similarly. The names of the operations are required to run sessions. In our toy example, the input operation is assigned the name "input" [(look here in the code)](https://github.com/anshuman23/tensorflex/blob/master/examples/graphdef_create.py#L7) and the output operation is assigned the name "output" [(look here in the code)](https://github.com/anshuman23/tensorflex/blob/master/examples/graphdef_create.py#L10). 

    - Now in Tensorflex, the Inference would go something like this:

        - First load the graph and look to see all the operations are correct. You will see "input" and "output" somewhere as mentioned before.
        - Then create the input tensors. First create matrices to house the tensor data as well as it's dimensions. As an example, let's say we set our input to be a 3x3 tensor with the first inputs all 1.0, second row to be 2.0, and third to be 3.0. The tensor is a `float32` tensor created using the `float32_tensor` function.    
        - Now to create the output tensor, since we know the matrix operations, the output will be a 3x2 tensor. We set the dimensions appropriately. Moreover, since we do not yet have the output values (we can only get them after the session is run), we will use the `float32_tensor_alloc` function instead of `float32_tensor`.
        - Finally, we need to run a session to get the output answers from sending the input tensors through the graph. We can see that the answer we get is exactly what we get when we do the matrix multiplications of the inputs with the weights and the addition of the biases. Also, the `run_session` function takes 5 inputs: the graph definition, the input tensor, the output tensor, the name of the input operation and the output operation. This is why knowing the names of your input and output operations is important.

    ```elixir
    iex(1)> {:ok, graph} = Tensorflex.read_graph "graphdef.pb"
    2018-06-04 00:32:53.993446: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that         this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
    {:ok, #Reference<0.1321508712.421658628.225797>}

    iex(2)> Tensorflex.get_graph_ops graph
    ["biases", "biases/read", "weights", "weights/read", "input", "MatMul", "add",
    "output"]

    iex(3)> in_vals = Tensorflex.create_matrix(3,3,[[1.0,1.0,1.0],[2.0,2.0,2.0],[3.0,3.0,3.0]])
    #Reference<0.1321508712.421658628.225826>

    iex(4)> in_dims = Tensorflex.create_matrix(1,2,[[3,3]]) 
    #Reference<0.1321508712.421658628.225834>

    iex(5)> {:ok, input_tensor} = Tensorflex.float32_tensor(in_vals, in_dims)
    {:ok, #Reference<0.1321508712.421658628.225842>}
    
    iex(6)> out_dims = Tensorflex.create_matrix(1,2,[[3,2]])
    #Reference<0.1321508712.421658628.225850>

    iex(7)> {:ok, output_tensor} = Tensorflex.float32_tensor_alloc(out_dims)       
    {:ok, #Reference<0.1321508712.421658628.225858>}
    
    iex(8)> Tensorflex.run_session(graph, input_tensor, output_tensor, "input", "output") 
    [
    [56.349998474121094, 39.26000213623047],
    [109.69999694824219, 75.52000427246094],
    [163.04998779296875, 111.77999877929688]
    ]
    ```

    
    
### Pull Requests Made 
- In chronological order:
    - [PR #2: Renamed app to Tensorflex from TensorflEx](https://github.com/anshum]()an23/tensorflex/pull/2) 
    - [PR #3: Added support for reading pretrained graph definition files](https://github.com/anshuman23/tensorflex/pull/3)
    - [PR #4: Merged all support functions into read_graph; returning atoms](https://github.com/anshuman23/tensorflex/pull/4)
    - [PR #6: Returning list of op names in get_graph_ops and extended error atoms to all TF error codes](https://github.com/anshuman23/tensorflex/pull/6)
    - [PR #7: Added tensor support for strings and for getting TF_DataType](https://github.com/anshuman23/tensorflex/pull/7)
    - [PR #8: Added matrix functions, numeral tensors, better returns & removed unnecessary POC code](https://github.com/anshuman23/tensorflex/pull/8)
    - [PR #9: Added freeze graph Python example](https://github.com/anshuman23/tensorflex/pull/9)
    - [PR #10: Tensors of TF_FLOAT and TF_DOUBLE type both supported](https://github.com/anshuman23/tensorflex/pull/10)
    - [PR #11: Tensor allocations and Tensorflow Session support added](https://github.com/anshuman23/tensorflex/pull/11)
    - [PR #12: Fixed issue regarding printing of outputs](https://github.com/anshuman23/tensorflex/pull/12)
    - [PR #13: Added another example for blog post](https://github.com/anshuman23/tensorflex/pull/13)
    - [PR #14: Added graph file check before loading](https://github.com/anshuman23/tensorflex/pull/13)
    - [PR #15: Added tests](https://github.com/anshuman23/tensorflex/pull/15)
    - [PR #16: Add linker flags for MacOS](https://github.com/anshuman23/tensorflex/pull/16)
    - [PR #18: Added image loading capabilities (as 3D uint8 Tensors)](https://github.com/anshuman23/tensorflex/pull/18)
