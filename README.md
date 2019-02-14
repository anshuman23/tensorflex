# Tensorflex

The paper detailing Tensorflex was presented at NeurIPS/NIPS 2018 as part of the MLOSS workshop. The paper can be found [here](https://openreview.net/pdf?id=rkxeCt6VhX).  

[![Build Status](https://travis-ci.com/anshuman23/tensorflex.svg?branch=master)](https://travis-ci.com/anshuman23/tensorflex)
[![Hex](https://img.shields.io/hexpm/v/tensorflex.svg?style=flat)](https://hex.pm/packages/tensorflex/0.1.2)

## Contents
- [__How to run__](https://github.com/anshuman23/tensorflex/#how-to-run)
- [__Documentation__](https://github.com/anshuman23/tensorflex/#documentation)
- [__Examples__](https://github.com/anshuman23/tensorflex/#examples)
- [__Pull Requests Made__](https://github.com/anshuman23/tensorflex/#pull-requests-made) 



### How to run
- You need to have the Tensorflow C API installed. Look [here](https://www.tensorflow.org/install/install_c) for details.
- You also need the C library `libjpeg`. If you are using Linux or OSX, it should already be present on your machine, otherwise be sure to install (`brew install libjpeg` for OSX, and `sudo apt-get install libjpeg-dev` for Ubuntu). 
- Simply add Tensorflex to your list of dependencies in `mix.exs` and you are good to go!: 

```elixir
{:tensorflex, "~> 0.1.2"}
```
In case you want the latest development version use this:
```elixir
{:tensorflex, github: "anshuman23/tensorflex"}
```

### Documentation
Tensorflex contains three main structs which handle different datatypes. These are `%Graph`, `%Matrix` and `%Tensor`. `%Graph` type structs handle pre-trained graph models, `%Matrix` handles Tensorflex 2-D matrices, and `%Tensor` handles Tensorflow Tensor types. The official Tensorflow documentation is present [here](https://hexdocs.pm/tensorflex/Tensorflex.html) and do note that this README only briefly discusses Tensorflex functionalities.

- `read_graph/1`:
  - Used for loading a Tensorflow `.pb` graph model in Tensorflex.

  - Reads in a pre-trained Tensorflow protobuf (`.pb`) Graph model binary file.

  - Returns a tuple `{:ok, %Graph}`. 
  
  - `%Graph` is an internal Tensorflex struct which holds the name of the graph file and the binary definition data that is read in via the `.pb` file. 

- `get_graph_ops/1`:
  - Used for listing all the operations in a Tensorflow `.pb` graph.

  - Reads in a Tensorflex ```%Graph``` struct obtained from `read_graph/1`.

  - Returns a list of all the operation names (as strings) that populate the graph model.

- `create_matrix/3`:
  - Creates a 2-D Tensorflex matrix from custom input specifications.

  - Takes three input arguments: number of rows in matrix (`nrows`), number of columns in matrix (`ncols`), and a list of lists of the data that will form the matrix (`datalist`).

  - Returns a `%Matrix` Tensorflex struct type.

- `matrix_pos/3`:
  - Used for accessing an element of a Tensorflex matrix. 

  - Takes in three input arguments: a Tensorflex `%Matrix` struct matrix, and the row (`row`) and column (`col`) values of the required element in the matrix. Both `row` and `col` here are __NOT__ zero indexed.

  - Returns the value as float.

- `size_of_matrix/1`:
  - Used for obtaining the size of a Tensorflex matrix.

  - Takes a Tensorflex `%Matrix` struct matrix as input.

  - Returns a tuple `{nrows, ncols}` where `nrows` represents the number of rows of the matrix and `ncols` represents the number of columns of the matrix.

- `append_to_matrix/2`:
  - Appends a single row to the back of a Tensorflex matrix.

  - Takes a Tensorflex `%Matrix` matrix as input and a single row of data (with the same number of columns as the original matrix) as a list of lists (`datalist`) to append to the original matrix.

  - Returns the extended and modified `%Matrix` struct matrix.

- `matrix_to_lists/1`:
  - Converts a Tensorflex matrix (back) to a list of lists format.

  - Takes a Tensorflex `%Matrix` struct matrix as input.

  - Returns a list of lists representing the data stored in the matrix.

  - __NOTE__: If the matrix contains very high dimensional data, typically obtained from a function like `load_csv_as_matrix/2`, then it is not recommended to convert the matrix back to a list of lists format due to a possibility of memory errors.

- `float64_tensor/2`, `float32_tensor/2`, `int32_tensor/2`:
  - Creates a `TF_DOUBLE`, `TF_FLOAT`, or `TF_INT32` tensor from Tensorflex matrices containing the values and dimensions specified.

  - Takes two arguments: a `%Matrix` matrix (`matrix1`) containing the values the tensor should have and another `%Matrix` matrix (`matrix2`) containing the dimensions of the required tensor.

  - Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal Tensorflex struct type that is used for holding tensor data and type.

- `float64_tensor/1`, `float32_tensor/1`, `int32_tensor/1`, `string_tensor/1`:
  - Creates a `TF_DOUBLE`, `TF_FLOAT`, `TF_INT32`, or `TF_STRING` constant value one-dimensional tensor from the input value specified.

  - Takes in a float, int or string value (depending on function) as input.

  - Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal Tensorflex struct type that is used for holding tensor data and type.

- `float64_tensor_alloc/1`, `float32_tensor_alloc/1`, `int32_tensor_alloc/1`:
  - Allocates a `TF_DOUBLE`, `TF_FLOAT`, or `TF_INT32` tensor of specified dimensions.
  
  - This function is generally used to allocate output tensors that do not hold any value data yet, but _will_ after the session is run for Inference. Output tensors of the required dimensions are allocated and then passed to the `run_session/5` function to hold the output values generated as predictions.

  - Takes a Tensorflex `%Matrix` struct matrix as input.

  - Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal Tensorflex struct type that is used for holding the potential tensor data and type.
  
- `tensor_datatype/1`:
  - Used to get the datatype of a created tensor.

  - Takes in a `%Tensor` struct tensor as input.

  - Returns a tuple `{:ok, datatype}` where `datatype` is an atom representing the list of Tensorflow `TF_DataType` tensor datatypes. Click [here](https://github.com/anshuman23/tensorflex/blob/master/c_src/c_api.h#L98-L122) to view a list of all possible datatypes.

- `load_image_as_tensor/1`:
  - Loads `JPEG` images into Tensorflex directly as a `TF_UINT8` tensor of dimensions `image height x image width x number of color channels`.

  - This function is very useful if you wish to do image classification using Convolutional Neural Networks, or other Deep Learning Models. One of the most widely adopted and robust image classification models is the [Inception](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz) model by Google. It makes classifications on images from over a 1000 classes with highly accurate results. The `load_image_as_tensor/1` function is an essential component for the prediction pipeline of the Inception model (and for other similar image classification models) to work in Tensorflex.

  - Reads in the path to a `JPEG` image file (`.jpg` or `.jpeg`).

  - Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal Tensorflex struct type that is used for holding the tensor data and type. Here the created Tensor is a `uint8` tensor (`TF_UINT8`).

  - __NOTE__: For now, only 3 channel RGB `JPEG` color images can be passed as arguments. Support for grayscale images and other image formats such as `PNG` will be added in the future. 

- `loads_csv_as_matrix/2`:
  - Loads high-dimensional data from a `CSV` file as a Tensorflex 2-D matrix in a super-fast manner.

  - The `load_csv_as_matrix/2` function is very fast-- when compared with the Python based `pandas` library for data science and analysis' function `read_csv` on the `test.csv` file from MNIST Kaggle data ([source](https://www.kaggle.com/c/digit-recognizer/data)), the following execution times were obtained:
    - `read_csv`: `2.549233` seconds
    - `load_csv_as_matrix/2`: `1.711494` seconds

  - This function takes in 2 arguments: a path to a valid CSV file (`filepath`) and other optional arguments `opts`. These include whether or not a header needs to be discarded in the CSV, and what the delimiter type is. These are specified by passing in an atom `:true` or `:false` to the `header:` key, and setting a string value for the `delimiter:` key. By default, the header is considered to be present (`:true`) and the delimiter is set to `,`.

  - Returns a `%Matrix` Tensorflex struct type.
  
- `run_session/5`:
  - Runs a Tensorflow session to generate predictions for a given graph, input data, and required input/output operations.

  - This function is the final step of the Inference (prediction) pipeline and generates output for a given set of input data, a pre-trained graph model, and the specified input and output operations of the graph.

  - Takes in five arguments: a pre-trained Tensorflow graph `.pb` model read in from the `read_graph/1` function (`graph`), an input tensor with the dimensions and data required for the input operation of the graph to run (`tensor1`), an output tensor allocated with the right dimensions (`tensor2`), the name of the input operation of the graph that needs where the input data is fed (`input_opname`), and the output operation name in the graph where the outputs are obtained (`output_opname`). The input tensor is generally created from the matrices manually or using the `load_csv_as_matrix/2` function, and then passed through to one of the tensor creation functions. For image classification the `load_image_as_tensor/1` can also be used to create the input tensor from an image. The output tensor is created using the tensor allocation functions (generally containing `alloc` at the end of the function name).  

  - Returns a List of Lists (similar to the `matrix_to_lists/1` function) containing the generated predictions as per the output tensor dimensions.
  
- `add_scalar_to_matrix/2`:
  - Adds scalar value to matrix.
  
  - Takes two arguments: `%Matrix` matrix and scalar value (int or float)
  
  - Returns a `%Matrix` modified matrix.

- `subtract_scalar_from_matrix/2`:
  - Subtracts scalar value from matrix.

  - Takes two arguments: `%Matrix` matrix and scalar value (int or float)

  - Returns a `%Matrix` modified matrix.
  
  
- `multiply_matrix_with_scalar/2`:
  - Multiplies scalar value with matrix.

  - Takes two arguments: `%Matrix` matrix and scalar value (int or float)

  - Returns a `%Matrix` modified matrix.

- `divide_matrix_by_scalar/2`:
  - Divides matrix values by scalar.

  - Takes two arguments: `%Matrix` matrix and scalar value (int or float)

  - Returns a `%Matrix` modified matrix.

- `add_matrices/2`:
  - Adds two matrices of same dimensions together.

  - Takes in two `%Matrix` matrices as arguments.

  - Returns the resultant `%Matrix` matrix.

- `subtract_matrices/2`:
  - Subtracts `matrix2` from `matrix1`.

  - Takes in two `%Matrix` matrices as arguments.

  - Returns the resultant `%Matrix` matrix.

- `tensor_to_matrix/1`:
  - Converts the data stored in a 2-D tensor back to a 2-D matrix.

  - Takes in a single argument as a `%Tensor` tensor (any `TF_Datatype`).

  - Returns a `%Matrix` 2-D matrix.

  - __NOTE__: Tensorflex doesn't currently support 3-D matrices, and therefore
  tensors that are 3-D (such as created using the `load_image_as_tensor/1`
  function) cannot be converted back to a matrix, yet. Support for 3-D matrices
  will be added soon.

### Examples
Examples are generally added in full description on my blog [here](http://anshumanc.ml). A blog post covering how to do classification on the Iris Dataset is present [here](http://www.anshumanc.ml/gsoc/2018/06/14/gsoc/).

--- 

__INCEPTION CNN MODEL EXAMPLE__:

Here we will briefly touch upon how to use the Google V3 Inception pre-trained graph model to do image classficiation from over a 1000 classes. First, the Inception V3 model can be downloaded here: http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

After unzipping, see that it contains the graphdef .pb file (`classify_image_graphdef.pb`) which contains our graph definition, a test jpeg image that should identify/classify as a panda (`cropped_panda.pb`) and a few other files I will detail later.

Now for running this in Tensorflex first the graph is loaded:

```elixir
iex(1)> {:ok, graph} = Tensorflex.read_graph("classify_image_graph_def.pb")
2018-07-29 00:48:19.849870: W tensorflow/core/framework/op_def_util.cc:346] Op BatchNormWithGlobalNormalization is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
{:ok,
 %Tensorflex.Graph{
   def: #Reference<0.2597534446.2498625538.211058>,
   name: "classify_image_graph_def.pb"
 }}
```
Then the cropped_panda image is loaded using the new `load_image_as_tensor` function:
```elixir
iex(2)> {:ok, input_tensor} = Tensorflex.load_image_as_tensor("cropped_panda.jpg")
{:ok,
 %Tensorflex.Tensor{
   datatype: :tf_uint8,
   tensor: #Reference<0.2597534446.2498625538.211093>
 }}

```
Then create the output tensor which will hold out output vector values. For the inception model, the output is received as a 1008x1 tensor, as there are 1008 classes in the model:
```elixir
iex(3)> out_dims = Tensorflex.create_matrix(1,2,[[1008,1]])
%Tensorflex.Matrix{
  data: #Reference<0.2597534446.2498625538.211103>,
  ncols: 2,
  nrows: 1
}

iex(4)> {:ok, output_tensor} = Tensorflex.float32_tensor_alloc(out_dims)
{:ok,
 %Tensorflex.Tensor{
   datatype: :tf_float,
   tensor: #Reference<0.2597534446.2498625538.211116>
 }}
```
Then the output results are read into a list called `results`. Also, the input operation in the Inception model is `DecodeJpeg` and the output operation is `softmax`:
```elixir
iex(5)> results = Tensorflex.run_session(graph, input_tensor, output_tensor, "DecodeJpeg", "softmax")
2018-07-29 00:51:13.631154: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
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
iex(6)> max_prob = List.flatten(results) |> Enum.max
0.8849328756332397

iex(7)> Enum.find_index(results |> List.flatten, fn(x) -> x == max_prob end)
169
``` 

We can thus see that the class with the maximum probability predicted (__0.8849328756332397__) for the image is __169__. We will now find what the 169 label corresponds to. For this we can look back into the unzipped Inception folder, where there is a file called `imagenet_2012_challenge_label_map_proto.pbtxt`. On opening this file, we can find the string class identifier for the `169` class index. This is `n02510455` and is present on Line 1556 in the file. Finally, we need to match this string identifier to a set of identification labels by referring to the file `imagenet_synset_to_human_label_map.txt` file. Here we can see that corresponding to the string class `n02510455` the human labels are `giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca` (Line 3691 in the file). 

Thus, we have correctly identified the animal in the image as a panda using Tensorflex!

---
    
__RNN LSTM SENTIMENT ANALYSIS MODEL EXAMPLE__:

A brief idea of what this example entails:
- The Recurrent Neural Network utilizes Long-Short-Term-Memory (LSTM) cells for holding the state for the data flowing in through the network
- In this example, we utilize the LSTM network for sentiment analysis on movie reviews data in Tensorflex. The trained models are originally created as part of an online tutorial [(source)](https://www.oreilly.com/learning/perform-sentiment-analysis-with-lstms-using-tensorflow) and are present in a Github repository [here](https://github.com/adeshpande3/LSTM-Sentiment-Analysis).

To do sentiment analysis in Tensorflex however, we first need to do some preprocessing and prepare the graph model (`.pb`) as done multiple times before in other examples. For that, in the `examples/rnn-lstm-example` directory there are two scripts: `freeze.py` and `create_input_data.py`. Prior to explaining the working of these scripts you first need to download the original saved models as well as the datasets:
- For the model, download from [here](https://github.com/adeshpande3/LSTM-Sentiment-Analysis/raw/master/models.tar.gz) and then store all the 4 model files in the `examples/rnn-lstm-example/model` folder
- For the dataset, download from [here](https://github.com/adeshpande3/LSTM-Sentiment-Analysis/raw/master/training_data.tar.gz). After decompressing, we do not need all the files, just the 2 numpy binaries `wordsList.npy` and `wordVectors.npy`. These will be used to encode our text data into `UTF-8` encoding for feeding our RNN as input.

Now, for the Python two scripts: `freeze.py` and `create_input_data.py`:
- `freeze.py`: This is used to create our `pb` model from the Python saved checkpoints. Here we will use the downloaded Python checkpoints' model to create the `.pb` graph. Just running `python freeze.py` after putting the model files in the correct directory will do the trick. In the same `./model/` folder, you will now see a file called `frozen_model_lstm.pb`. This is the file which we will load into Tensorflex. In case for some reason you want to skip this step and just get the loaded graph here is a Dropbox [link](https://www.dropbox.com/s/xp1bphy0k40v5r6/frozen_model_lstm.pb?dl=0)
- `create_input_data.py`: Even if we can load our model into Tensorflex, we also need some data to do inference on. For that, we will write our own example sentences and convert them (read encode) to a numeral (`int32`) format that can be used by the network as input. For that, you can inspect the code in the script to get an understanding of what is happening. Basically, the neural network takes in an input of a `24x250` `int32` (matrix) tensor created from text which has been encoded as `UTF-8`. Again, running `python create_input_data.py` will give you two `csv` files (one indicating positive sentiment and the other a negative sentiment) which we will later load into Tensorflex. The two sentences converted are:
  - Negative sentiment sentence: _That movie was terrible._
  - Positive sentiment sentence: _That movie was the best one I have ever seen._

              
Both of these get converted to two files `inputMatrixPositive.csv` and `inputMatrixNegative.csv` (by `create_input_data.py`) which we load into Tensorflex next.

__Inference in Tensorflex:__
Now we do sentiment analysis in Tensorflex. A few things to note:
- The input graph operation is named `Placeholder_1`
- The output graph operation is named `add` and is the eventual result of a matrix multiplication. Of this obtained result we only need the first row
- Here the input is going to be a integer valued matrix tensor of dimensions `24x250` representing our sentence/review
- The output will have 2 columns, as there are 2 classes-- for positive and negative sentiment respectively. Since we will only be needing only the first row we will get our result in a `1x2` vector. If the value of the first column is higher than the second column, then the network indicates a positive sentiment otherwise a negative sentiment. All this can be observed in the original repository in a Jupyter notebook [here](https://github.com/adeshpande3/LSTM-Sentiment-Analysis):
```elixir
iex(1)> {:ok, graph} = Tensorflex.read_graph "examples/rnn-lstm-example/model/frozen_model_lstm.pb"
{:ok,
 %Tensorflex.Graph{
   def: #Reference<0.713975820.1050542081.11558>,
   name: "examples/rnn-lstm-example/model/frozen_model_lstm.pb"
 }}

iex(2)> Tensorflex.get_graph_ops graph
["Placeholder_1", "embedding_lookup/params_0", "embedding_lookup",
 "transpose/perm", "transpose", "rnn/Shape", "rnn/strided_slice/stack",
 "rnn/strided_slice/stack_1", "rnn/strided_slice/stack_2", "rnn/strided_slice",
 "rnn/stack/1", "rnn/stack", "rnn/zeros/Const", "rnn/zeros", "rnn/stack_1/1",
 "rnn/stack_1", "rnn/zeros_1/Const", "rnn/zeros_1", "rnn/Shape_1",
 "rnn/strided_slice_2/stack", "rnn/strided_slice_2/stack_1",
 "rnn/strided_slice_2/stack_2", "rnn/strided_slice_2", "rnn/time",
 "rnn/TensorArray", "rnn/TensorArray_1", "rnn/TensorArrayUnstack/Shape",
 "rnn/TensorArrayUnstack/strided_slice/stack",
 "rnn/TensorArrayUnstack/strided_slice/stack_1",
 "rnn/TensorArrayUnstack/strided_slice/stack_2",
 "rnn/TensorArrayUnstack/strided_slice", "rnn/TensorArrayUnstack/range/start",
 "rnn/TensorArrayUnstack/range/delta", "rnn/TensorArrayUnstack/range",
 "rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3",
 "rnn/while/Enter", "rnn/while/Enter_1", "rnn/while/Enter_2",
 "rnn/while/Enter_3", "rnn/while/Merge", "rnn/while/Merge_1",
 "rnn/while/Merge_2", "rnn/while/Merge_3", "rnn/while/Less/Enter",
 "rnn/while/Less", "rnn/while/LoopCond", "rnn/while/Switch",
 "rnn/while/Switch_1", "rnn/while/Switch_2", "rnn/while/Switch_3", ...]
```
First we will try for positive sentiment:
```elixir
iex(3)> input_vals = Tensorflex.load_csv_as_matrix("examples/rnn-lstm-example/inputMatrixPositive.csv", header: :false)
%Tensorflex.Matrix{
  data: #Reference<0.713975820.1050542081.13138>,
  ncols: 250,
  nrows: 24
}

iex(4)> input_dims = Tensorflex.create_matrix(1,2,[[24,250]])
%Tensorflex.Matrix{
  data: #Reference<0.713975820.1050542081.13575>,
  ncols: 2,
  nrows: 1
}

iex(5)> {:ok, input_tensor} = Tensorflex.int32_tensor(input_vals, input_dims)
{:ok,
 %Tensorflex.Tensor{
   datatype: :tf_int32,
   tensor: #Reference<0.713975820.1050542081.14434>
 }}

iex(6)> output_dims = Tensorflex.create_matrix(1,2,[[24,2]])
%Tensorflex.Matrix{
  data: #Reference<0.713975820.1050542081.14870>,
  ncols: 2,
  nrows: 1
}

iex(7)> {:ok, output_tensor} = Tensorflex.float32_tensor_alloc(output_dims)
{:ok,
 %Tensorflex.Tensor{
   datatype: :tf_float,
   tensor: #Reference<0.713975820.1050542081.15363>
 }}
```
We only need the first row, the rest do not indicate anything:
```elixir
iex(8)> [result_pos | _ ] = Tensorflex.run_session(graph, input_tensor,output_tensor, "Placeholder_1", "add")
[
  [4.483788013458252, -1.273943305015564],
  [-0.17151066660881042, -2.165886402130127],
  [0.9569928646087646, -1.131581425666809],
  [0.5669126510620117, -1.3842089176177979],
  [-1.4346938133239746, -4.0750861167907715],
  [0.4680981934070587, -1.3494354486465454],
  [1.068990707397461, -2.0195648670196533],
  [3.427264451980591, 0.48857203125953674],
  [0.6307879686355591, -2.069119691848755],
  [0.35061028599739075, -1.700657844543457],
  [3.7612719535827637, 2.421398878097534],
  [2.7635951042175293, -0.7214710116386414],
  [1.146680235862732, -0.8688814640045166],
  [0.8996094465255737, -1.0183486938476563],
  [0.23605018854141235, -1.893072247505188],
  [2.8790698051452637, -0.37355837225914],
  [-1.7325369119644165, -3.6470277309417725],
  [-1.687785029411316, -4.903762340545654],
  [3.6726789474487305, 0.14170047640800476],
  [0.982108473777771, -1.554244875907898],
  [2.248904228210449, 1.0617655515670776],
  [0.3663095533847809, -3.5266385078430176],
  [-1.009346604347229, -2.901120901107788],
  [3.0659966468811035, -1.7605335712432861]
]

iex(9)> result_pos
[4.483788013458252, -1.273943305015564]
```

Thus we can clearly see that the RNN predicts a positive sentiment. For a negative sentiment, next:

```elixir
iex(10)> input_vals = Tensorflex.load_csv_as_matrix("examples/rnn-lstm-example/inputMatrixNegative.csv", header: :false)
%Tensorflex.Matrix{
  data: #Reference<0.713975820.1050542081.16780>,
  ncols: 250,
  nrows: 24
}

iex(11)> {:ok, input_tensor} = Tensorflex.int32_tensor(input_vals,input_dims)
{:ok,              
 %Tensorflex.Tensor{
   datatype: :tf_int32,
   tensor: #Reference<0.713975820.1050542081.16788>
 }}

iex(12)> [result_neg|_] = Tensorflex.run_session(graph, input_tensor,output_tensor, "Placeholder_1", "add")
[
  [0.7635725736618042, 10.895986557006836],
  [2.205151319503784, -0.6267685294151306],
  [3.5995595455169678, -0.1240251287817955],
  [-1.6063352823257446, -3.586883068084717],
  [1.9608432054519653, -3.084211826324463],
  [3.772461414337158, -0.19421455264091492],
  [3.9185996055603027, 0.4442034661769867],
  [3.010765552520752, -1.4757057428359985],
  [3.23650860786438, -0.008513949811458588],
  [2.263028144836426, -0.7358709573745728],
  [0.206748828291893, -2.1945853233337402],
  [2.913491725921631, 0.8632720708847046],
  [0.15935257077217102, -2.9757845401763916],
  [-0.7757357358932495, -2.360766649246216],
  [3.7359719276428223, -0.7668198347091675],
  [2.2896337509155273, -0.45704856514930725],
  [-1.5497230291366577, -4.42919921875],
  [-2.8478822708129883, -5.541027545928955],
  [1.894787073135376, -0.8441318273544312],
  [0.15720489621162415, -2.699129819869995],
  [-0.18114641308784485, -2.988100051879883],
  [3.342879056930542, 2.1714375019073486],
  [2.906526565551758, 0.18969044089317322],
  [0.8568912744522095, -1.7559258937835693]
]
iex(13)> result_neg
[0.7635725736618042, 10.895986557006836]
```
Thus we can clearly see that in this case the RNN indicates negative sentiment! Our model works!
    
### Pull Requests Made 
- In chronological order:
    - [PR #2: Renamed app to Tensorflex from TensorflEx](https://github.com/anshuman23/tensorflex/pull/2) 
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
    - [PR #19: Wrapped references around structs](https://github.com/anshuman23/tensorflex/pull/19)
    - [PR #20: Moved basic C error checking to Elixir](https://github.com/anshuman23/tensorflex/pull/20)
    - [PR #21: Added Keras example](https://github.com/anshuman23/tensorflex/pull/21)
    - [PR #22: Added "append" function for matrices](https://github.com/anshuman23/tensorflex/pull/22)
    - [PR #23: Added fast C based direct CSV-to-matrix functionality with options](https://github.com/anshuman23/tensorflex/pull/23)
    - [PR #24: Added TF_INT32 tensors and tensor allocators](https://github.com/anshuman23/tensorflex/pull/24)
    - [PR #25: Added RNN (LSTM) example](https://github.com/anshuman23/tensorflex/pull/25)
    - [PR #26: Added documentation](https://github.com/anshuman23/tensorflex/pull/26)
    - [PR #28: Added improved tests](https://github.com/anshuman23/tensorflex/pull/28)
    - [PR #29: Adding metadata to mix.exs](https://github.com/anshuman23/tensorflex/pull/29)
    - [PR #31: Update nifs.ex](https://github.com/anshuman23/tensorflex/pull/31)
    - [PR #32: Fixed indentation and corrected warnings](https://github.com/anshuman23/tensorflex/pull/32)
    - [PR #35: Added new matrix operations](https://github.com/anshuman23/tensorflex/pull/35)
    - [PR #36: Fixed bugs in C code](https://github.com/anshuman23/tensorflex/pull/36)
    - [PR #37: Added tensor_to_matrix/1 (with tests/docs)](https://github.com/anshuman23/tensorflex/pull/37)
    - [PR #38: Create CONTRIBUTING.md](https://github.com/anshuman23/tensorflex/pull/38)
    - [PR #39: Formatted C code](https://github.com/anshuman23/tensorflex/pull/39)
