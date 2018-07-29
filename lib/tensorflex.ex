defmodule Tensorflex do
  @moduledoc """
  A simple and fast library for running Tensorflow graph models in Elixir.
Tensorflex is written around the [Tensorflow C
API](https://www.tensorflow.org/install/install_c), and allows Elixir
developers to leverage Machine Learning and Deep Learning solutions in their
projects.

  __NOTE__:

  - Make sure that the C API version and Python API version (assuming you are
    using the Python API for first training your models) are the latest. As of
July 2018, the latest version is `r1.9`.

  - Since Tensorflex provides Inference capability for pre-trained graph
    models, it is assumed you have adequate knowledge of the pre-trained models
you are using (such as the input data type/dimensions, input and output
operation names, etc.). Some basic understanding of the [Tensorflow Python
API](https://www.tensorflow.org/api_docs/python/) can come in very handy. 

  - Tensorflex consists of multiple NIFs, so exercise caution while using it--
    providing incorrect operation names for running sessions, incorrect
dimensions of tensors than the actual pre-trained graph requires, providing
different tensor datatypes than the ones required by the graph can all lead to
failure. While these are not easy errors to make, do ensure that you test your
solution well before deployment.  
"""
  
  alias Tensorflex.{NIFs, Graph, Tensor, Matrix}

  defp empty_list?([[]]), do: true
  defp empty_list?(list) when is_list(list) do
    false
  end

  @doc """
  Used for loading a Tensorflow `.pb` graph model in Tensorflex.

  Reads in a pre-trained Tensorflow protobuf (`.pb`) Graph model binary file.

  Returns a tuple `{:ok, %Graph}`. 
  
  `%Graph` is an internal Tensorflex struct which holds the name of the graph
file and the binary definition data that is read in via the `.pb` file. 

  ## Examples:
  
  _Reading in a graph_

  As an example, we can try reading in the
[Inception](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz)
convolutional neural network based image classification graph model by Google.
The graph file is named `classify_image_graph_def.pb`: 
```elixir
  iex(1)> {:ok, graph} = Tensorflex.read_graph "classify_image_graph_def.pb"
  2018-07-23 15:31:35.949345: W tensorflow/core/framework/op_def_util.cc:346] Op BatchNormWithGlobalNormalization is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
  {:ok,
  %Tensorflex.Graph{
    def: #Reference<0.3018278404.759824385.5268>,
    name: "classify_image_graph_def.pb"
  }}
  ```
  Generally to check that the loaded graph model is correct and contains
computational operations, the `get_graph_ops/1` function is useful: 
```elixir
  iex(2)> Tensorflex.get_graph_ops graph
  ["DecodeJpeg/contents", "DecodeJpeg", "Cast", "ExpandDims/dim", "ExpandDims",
   "ResizeBilinear/size", "ResizeBilinear", "Sub/y", "Sub", "Mul/y", "Mul",
   "conv/conv2d_params", "conv/Conv2D", "conv/batchnorm/beta",
   "conv/batchnorm/gamma", "conv/batchnorm/moving_mean",
   "conv/batchnorm/moving_variance", "conv/batchnorm", "conv/CheckNumerics",
   "conv/control_dependency", "conv", "conv_1/conv2d_params", "conv_1/Conv2D",
   "conv_1/batchnorm/beta", "conv_1/batchnorm/gamma",
   "conv_1/batchnorm/moving_mean", "conv_1/batchnorm/moving_variance",
   "conv_1/batchnorm", "conv_1/CheckNumerics", "conv_1/control_dependency",
   "conv_1", "conv_2/conv2d_params", "conv_2/Conv2D", "conv_2/batchnorm/beta",
   "conv_2/batchnorm/gamma", "conv_2/batchnorm/moving_mean",
   "conv_2/batchnorm/moving_variance", "conv_2/batchnorm", "conv_2/CheckNumerics",
   "conv_2/control_dependency", "conv_2", "pool/CheckNumerics",
   "pool/control_dependency", "pool", "conv_3/conv2d_params", "conv_3/Conv2D",
   "conv_3/batchnorm/beta", "conv_3/batchnorm/gamma",
   "conv_3/batchnorm/moving_mean", "conv_3/batchnorm/moving_variance", ...]
  
  ```

  _Incorrect usage will `raise`_:

  ```elixir
  iex(3)> {:ok, graph} = Tensorflex.read_graph "Makefile"
  ** (ArgumentError) file is not a protobuf .pb file
  (tensorflex) lib/tensorflex.ex:27: Tensorflex.read_graph/1

  iex(3)> {:ok, graph} = Tensorflex.read_graph "Makefile.pb"
  ** (ArgumentError) graph definition file does not exist
  (tensorflex) lib/tensorflex.ex:23: Tensorflex.read_graph/1
  
  ```
  """
  
  def read_graph(filepath) do
    unless File.exists?(filepath) do
      raise ArgumentError, "graph definition file does not exist"
    end

    unless (Path.extname(filepath) == ".pb") do
      raise ArgumentError, "file is not a protobuf .pb file"
    end
     
    {:ok, ref} = NIFs.read_graph(filepath)
    {:ok, %Graph{def: ref, name: filepath}}
  end

  @doc """
  Used for listing all the operations in a Tensorflow `.pb` graph.

  Reads in a Tensorflex ```%Graph``` struct obtained from `read_graph/1`.

  Returns a list of all the operation names (as strings) that populate the
graph model.

  ## Examples

  - _Google Inception CNN Model_
    ([source](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz))
  
  ```elixir
  iex(1)> {:ok, graph} = Tensorflex.read_graph "classify_image_graph_def.pb"
  2018-07-23 15:31:35.949345: W tensorflow/core/framework/op_def_util.cc:346] Op BatchNormWithGlobalNormalization is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
  {:ok,
  %Tensorflex.Graph{
    def: #Reference<0.3018278404.759824385.5268>,
    name: "classify_image_graph_def.pb"
  }}

  iex(2)> Tensorflex.get_graph_ops graph
  ["DecodeJpeg/contents", "DecodeJpeg", "Cast", "ExpandDims/dim", "ExpandDims",
   "ResizeBilinear/size", "ResizeBilinear", "Sub/y", "Sub", "Mul/y", "Mul",
   "conv/conv2d_params", "conv/Conv2D", "conv/batchnorm/beta",
   "conv/batchnorm/gamma", "conv/batchnorm/moving_mean",
   "conv/batchnorm/moving_variance", "conv/batchnorm", "conv/CheckNumerics",
   "conv/control_dependency", "conv", "conv_1/conv2d_params", "conv_1/Conv2D",
   "conv_1/batchnorm/beta", "conv_1/batchnorm/gamma",
   "conv_1/batchnorm/moving_mean", "conv_1/batchnorm/moving_variance",
   "conv_1/batchnorm", "conv_1/CheckNumerics", "conv_1/control_dependency",
   "conv_1", "conv_2/conv2d_params", "conv_2/Conv2D", "conv_2/batchnorm/beta",
   "conv_2/batchnorm/gamma", "conv_2/batchnorm/moving_mean",
   "conv_2/batchnorm/moving_variance", "conv_2/batchnorm", "conv_2/CheckNumerics",
   "conv_2/control_dependency", "conv_2", "pool/CheckNumerics",
   "pool/control_dependency", "pool", "conv_3/conv2d_params", "conv_3/Conv2D",
   "conv_3/batchnorm/beta", "conv_3/batchnorm/gamma",
   "conv_3/batchnorm/moving_mean", "conv_3/batchnorm/moving_variance", ...]
  ```

  - _Iris Dataset MLP Model_
    ([source](http://www.anshumanc.ml/gsoc/2018/06/14/gsoc/))

  ```elixir
  iex(1)> {:ok, graph} = Tensorflex.read_graph "graphdef_iris.pb"
  {:ok,
  %Tensorflex.Graph{
    def: #Reference<0.4109712726.1847984130.24506>,
    name: "graphdef_iris.pb"
  }}

  iex(2)> Tensorflex.get_graph_ops graph
  ["input", "weights1", "weights1/read", "biases1", "biases1/read", "weights2", "weights2/read", "biases2", "biases2/read", "MatMul", "Add", "Relu", "MatMul_1", "Add_1", "output"]
  
  ```

  - _Toy Computational Graph Model_
    ([source](https://github.com/anshuman23/tensorflex/tree/master/examples/toy-example))

  ```elixir
  iex(1)> {:ok, graph} = Tensorflex.read_graph "graphdef_toy.pb"
  {:ok,
  %Tensorflex.Graph{
    def: #Reference<0.1274892327.1580335105.235135>,
    name: "graphdef_toy.pb"
  }}

  iex(2)> Tensorflex.get_graph_ops graph
  ["input", "weights", "weights/read", "biases", "biases/read", "MatMul", "add", "output"]
  ```

  - _RNN LSTM Sentiment Analysis Model_
    ([source](https://github.com/anshuman23/tensorflex/pull/25))
  
  ```elixir
  iex(1)> {:ok, graph} = Tensorflex.read_graph "frozen_model_lstm.pb"
  {:ok,
  %Tensorflex.Graph{
    def: #Reference<0.713975820.1050542081.11558>,
    name: "frozen_model_lstm.pb"
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
  """

  def get_graph_ops(%Graph{def: ref, name: filepath}) do
    NIFs.get_graph_ops(ref)
  end

  @doc """
  Creates a 2-D Tensorflex matrix from custom input specifications.

  Takes three input arguments: number of rows in matrix (`nrows`), number of
columns in matrix (`ncols`), and a list of lists of the data that will form the
matrix (`datalist`).

  Returns a `%Matrix` Tensorflex struct type.

  ## Examples:
 
  _Creating a new matrix_
  
  ```elixir
  iex(1)> mat = Tensorflex.create_matrix(2,3,[[2.2,1.3,44.5],[5.5,6.1,3.333]])    
  %Tensorflex.Matrix{
    data: #Reference<0.759278808.823525378.128525>,
    ncols: 3,
    nrows: 2
  }
  ```

  All `%Matrix` Tensorflex matrices can be passed in to the other matrix
inspection and manipulation functions-- `matrix_pos/3`,`size_of_matrix/1`,
`matrix_to_lists/1`, and `append_to_matrix/2`:

  ```elixir
  iex(1)> mat = Tensorflex.create_matrix(4,4,[[123,431,23,1],[1,2,3,4],[5,6,7,8],[768,564,44,5]])
  %Tensorflex.Matrix{
    data: #Reference<0.878138179.2435973124.131489>,
    ncols: 4,
    nrows: 4
  }

  iex(2)> mat = Tensorflex.append_to_matrix(mat, [[1,1,1,1]])
  %Tensorflex.Matrix{
    data: #Reference<0.878138179.2435973124.131489>,
    ncols: 4,
    nrows: 5
  }

  iex(3)> Tensorflex.matrix_to_lists mat
  [
    [123.0, 431.0, 23.0, 1.0],
    [1.0, 2.0, 3.0, 4.0],
    [5.0, 6.0, 7.0, 8.0],
    [768.0, 564.0, 44.0, 5.0],
    [1.0, 1.0, 1.0, 1.0]
  ]

  iex(4)> Tensorflex.matrix_pos(mat,5,3)
  1.0

  iex(5)> Tensorflex.size_of_matrix mat
  {5, 4}
  ```

  _Incorrect usage will `raise`_:

  ```elixir
  iex(1)> Tensorflex.create_matrix(1,2,[[1,2,3]])
  ** (ArgumentError) argument error
  (tensorflex) Tensorflex.NIFs.create_matrix(1, 2, [[1, 2, 3]])
  (tensorflex) lib/tensorflex.ex:247: Tensorflex.create_matrix/3

  iex(1)> Tensorflex.create_matrix(2,1,[[1,2,3]])
  ** (ArgumentError) argument error
  (tensorflex) Tensorflex.NIFs.create_matrix(2, 1, [[1, 2, 3]])
  (tensorflex) lib/tensorflex.ex:247: Tensorflex.create_matrix/3
  
  iex(1)> Tensorflex.create_matrix(2,3,[[1.1,23,3.4], []])
  ** (ArgumentError) argument error
    (tensorflex) Tensorflex.NIFs.create_matrix(2, 3, [[1.1, 23, 3.4], []])
    (tensorflex) lib/tensorflex.ex:247: Tensorflex.create_matrix/3
    
  iex(1)> Tensorflex.create_matrix(1,2,[[]])              
  ** (ArgumentError) data provided cannot be an empty list
  (tensorflex) lib/tensorflex.ex:243: Tensorflex.create_matrix/3
  
  iex(1)> Tensorflex.create_matrix(-1,2,[[3,4]])
  ** (FunctionClauseError) no function clause matching in Tensorflex.create_matrix/3    
  ```
  """

  def create_matrix(nrows, ncols, datalist) when nrows > 0 and ncols > 0 do
    if(empty_list? datalist) do
      raise ArgumentError, "data provided cannot be an empty list"
    end
    
    ref = NIFs.create_matrix(nrows, ncols, datalist)
    %Matrix{nrows: nrows, ncols: ncols, data: ref}
  end

  @doc """

  Used for accessing an element of a Tensorflex matrix. 

  Takes in three input arguments: a Tensorflex `%Matrix` struct matrix, and the
row (`row`) and column (`col`) values of the required element in the matrix.
Both `row` and `col` here are __NOT__ zero indexed.

  Returns the value as float.

  ## Examples

  ```elixir
  iex(1)> mat = Tensorflex.create_matrix(2,3,[[2.2,1.3,44.5],[5.5,6.1,3.333]])
  %Tensorflex.Matrix{
    data: #Reference<0.759278808.823525378.128525>,
    ncols: 3,
    nrows: 2
  }

  iex(2)> Tensorflex.matrix_pos(mat,2,1)
  5.5

  iex(3)> Tensorflex.matrix_pos(mat,1,3)
  44.5
  
  ```
  """

  def matrix_pos(%Matrix{nrows: nrows, ncols: ncols, data: ref}, row, col) when row > 0 and col > 0 do
    NIFs.matrix_pos(ref, row, col)
  end

  @doc """
  Used for obtaining the size of a Tensorflex matrix.

  Takes a Tensorflex `%Matrix` struct matrix as input.

Returns a tuple `{nrows, ncols}` where `nrows` represents the number of rows of
the matrix and `ncols` represents the number of columns of the matrix.

  ## Examples

  ```elixir
  iex(1)> mat = Tensorflex.create_matrix(2,3,[[2.2,1.3,44.5],[5.5,6.1,3.333]])
  %Tensorflex.Matrix{
    data: #Reference<0.759278808.823525378.128525>,
    ncols: 3,
    nrows: 2
  }
  
  iex(2)> Tensorflex.size_of_matrix mat
  {2, 3}
  ```
  """

  def size_of_matrix(%Matrix{nrows: nrows, ncols: ncols, data: ref}) do
    {nrows, ncols}
  end

  @doc """
  Appends a single row to the back of a Tensorflex matrix.

  Takes a Tensorflex `%Matrix` matrix as input and a single row of data (with
the same number of columns as the original matrix) as a list of lists
(`datalist`) to append to the original matrix.

  Returns the extended and modified `%Matrix` struct matrix.

  ## Examples

  ```elixir
  iex(1)> m = Tensorflex.create_matrix(2,3,[[23,23,23],[32,32,32]])
  %Tensorflex.Matrix{
    data: #Reference<0.153563642.2042232833.193025>,
    ncols: 3,
    nrows: 2
  }

  iex(2)> m = Tensorflex.append_to_matrix(m,[[2,2,2]])
  %Tensorflex.Matrix{
    data: #Reference<0.153563642.2042232833.193025>,
    ncols: 3,
    nrows: 3
  }

  iex(3)> m = Tensorflex.append_to_matrix(m,[[3,3,3]])
  %Tensorflex.Matrix{
    data: #Reference<0.153563642.2042232833.193025>,
    ncols: 3,
    nrows: 4
  }

  iex(4)> m |> Tensorflex.matrix_to_lists
  [[23.0, 23.0, 23.0], [32.0, 32.0, 32.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
  
  ```
  _Incorrect usage will `raise`_:

  ```elixir
  iex(5)> m = Tensorflex.append_to_matrix(m,[[2,2,2],[3,3,3]])
  ** (ArgumentError) data columns must be same as matrix and number of rows must be 1
  (tensorflex) lib/tensorflex.ex:345: Tensorflex.append_to_matrix/2
  
  iex(5)> m = Tensorflex.append_to_matrix(m,[[2,2,2,2]])      
  ** (ArgumentError) data columns must be same as matrix and number of rows must be 1
  (tensorflex) lib/tensorflex.ex:345: Tensorflex.append_to_matrix/2
  ```
  
  """

  def append_to_matrix(%Matrix{nrows: nrows, ncols: ncols, data: ref}, datalist) do
    unless (datalist |> List.flatten |> Kernel.length) == ncols do
	raise ArgumentError, "data columns must be same as matrix and number of rows must be 1"
    end
    new_ref = NIFs.append_to_matrix(ref, datalist)
    %Matrix{nrows: nrows+1, ncols: ncols, data: new_ref}
  end

  @doc """
  Converts a Tensorflex matrix (back) to a list of lists format.

  Takes a Tensorflex `%Matrix` struct matrix as input.

  Returns a list of lists representing the data stored in the matrix.

  __NOTE__: If the matrix contains very high dimensional data, typically
obtained from a function like `load_csv_as_matrix/2`, then it is not
recommended to convert the matrix back to a list of lists format due to a
possibility of memory errors.

  ## Examples

  ```elixir
  iex(1)> Tensorflex.create_matrix(2,3,[[23,23,23],[32,32,32]]) |> Tensorflex.matrix_to_lists
  [[23.0, 23.0, 23.0], [32.0, 32.0, 32.0]]
  ```
  """

  def matrix_to_lists(%Matrix{nrows: nrows, ncols: ncols, data: ref}) do
    NIFs.matrix_to_lists(ref)
  end

  @doc """
  Creates a `TF_DOUBLE` tensor from Tensorflex matrices containing the values
and dimensions specified.

  Takes two arguments: a `%Matrix` matrix (`matrix1`) containing the values the
tensor should have and another `%Matrix` matrix (`matrix2`) containing the
dimensions of the required tensor.

  Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal
Tensorflex struct type that is used for holding tensor data and type.

  ## Examples:

  ```elixir
  iex(1)> vals = Tensorflex.create_matrix(2,3,[[12.0,45.2,2.11],[36.7,8.09,9.81]])
  %Tensorflex.Matrix{
    data: #Reference<0.1251941183.3671982081.254268>,
    ncols: 3,
    nrows: 2
  }

  iex(2)> dims = Tensorflex.create_matrix(1,2,[[2,3]])
  %Tensorflex.Matrix{
    data: #Reference<0.1251941183.3671982081.254723>,
    ncols: 2,
    nrows: 1
  }

  iex(3)> {:ok, tensor} = Tensorflex.float64_tensor vals,dims
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_double,
    tensor: #Reference<0.1251941183.3671982081.255216>
  }}

  ```
  """

  def float64_tensor(%Matrix{nrows: val_rows, ncols: val_cols, data: val_ref}, %Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = NIFs.float64_tensor(val_ref, dim_ref)
    {:ok, %Tensor{datatype: :tf_double, tensor: ref}} 
  end

  @doc """
  Creates a `TF_DOUBLE` constant value one-dimensional tensor from the floating
point value specified.

  Takes in a float value as input.

  Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal
Tensorflex struct type that is used for holding tensor data and type.

  ## Examples

  ```elixir
  iex(1)> {:ok, tensor} = Tensorflex.float64_tensor 123.123
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_double,
    tensor: #Reference<0.2778616536.4219338753.155412>
  }}
  
  ```
  
  _Incorrect usage will `raise`_:

  ```elixir
  iex(2)> {:ok, tensor} = Tensorflex.float64_tensor "123.123"
  ** (FunctionClauseError) no function clause matching in Tensorflex.float64_tensor/1
  
  iex(2)> {:ok, tensor} = Tensorflex.float64_tensor 123      
  ** (FunctionClauseError) no function clause matching in Tensorflex.float64_tensor/1
  ```
  """
  
  def float64_tensor(floatval) when is_float(floatval) do
    {:ok, ref} = NIFs.float64_tensor(floatval)
    {:ok, %Tensor{datatype: :tf_double, tensor: ref}}
  end

  @doc """
  Creates a `TF_FLOAT` tensor from Tensorflex matrices containing the values
and dimensions specified.

  Takes two arguments: a `%Matrix` matrix (`matrix1`) containing the values the
tensor should have and another `%Matrix` matrix (`matrix2`) containing the
dimensions of the required tensor.

Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal
Tensorflex struct type that is used for holding tensor data and type.

  ## Examples:

  ```elixir
  iex(1)> vals = Tensorflex.create_matrix(2,3,[[12.0,45.2,2.11],[36.7,8.09,9.81]])
  %Tensorflex.Matrix{
    data: #Reference<0.1251941183.3671982081.254268>,
    ncols: 3,
    nrows: 2
  }
  
  iex(2)> dims = Tensorflex.create_matrix(1,2,[[2,3]])
  %Tensorflex.Matrix{
    data: #Reference<0.1251941183.3671982081.254723>,
    ncols: 2,
    nrows: 1
  }
    
  iex(3)> {:ok, tensor} = Tensorflex.float32_tensor vals,dims
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_float,
    tensor: #Reference<0.1251941183.3671982081.255228>
  }}
  
  ```
  """

  def float32_tensor(%Matrix{nrows: val_rows, ncols: val_cols, data: val_ref}, %Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = NIFs.float32_tensor(val_ref, dim_ref)
    {:ok, %Tensor{datatype: :tf_float, tensor: ref}} 
  end

  @doc """
  Creates a `TF_FLOAT` constant value one-dimensional tensor from the floating
point value specified.
  
  Takes in a float value as input.
  
  Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal
Tensorflex struct type that is used for holding tensor data and type.

  ## Examples

  ```elixir
  iex(1)> {:ok, tensor} = Tensorflex.float32_tensor 123.123
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_float,
    tensor: #Reference<0.2011963375.1804468228.236110>
  }}
  
  ```
  
  _Incorrect usage will `raise`_:

  ```elixir
  iex(2)> {:ok, tensor} = Tensorflex.float32_tensor "123.123"
  ** (FunctionClauseError) no function clause matching in Tensorflex.float32_tensor/1 

  iex(2)> {:ok, tensor} = Tensorflex.float32_tensor 123      
  ** (FunctionClauseError) no function clause matching in Tensorflex.float32_tensor/1
  ```
  """

  def float32_tensor(floatval) when is_float(floatval) do
    {:ok, ref} = NIFs.float32_tensor(floatval)
    {:ok, %Tensor{datatype: :tf_float, tensor: ref}}
  end

  @doc """
  Creates a `TF_INT32` tensor from Tensorflex matrices containing the values
and dimensions specified.
  
  Takes two arguments: a `%Matrix` matrix (`matrix1`) containing the values the
tensor should have and another `%Matrix` matrix (`matrix2`) containing the
dimensions of the required tensor.

Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal
Tensorflex struct type that is used for holding tensor data and type.

  __NOTE__: In case floating point values are passed in the values matrix
(`matrix1`) as arguments for this function, the tensor will still be created
and all the float values will be typecast to integers. 

  ## Examples:

  ```elixir
  iex(1)> vals = Tensorflex.create_matrix(2,3,[[123,45,333],[2,2,899]]) 
  %Tensorflex.Matrix{
    data: #Reference<0.1256144000.2868510721.170449>,
    ncols: 3,
    nrows: 2
  }
  
  iex(2)> dims = Tensorflex.create_matrix(1,2,[[2,3]])
  %Tensorflex.Matrix{
    data: #Reference<0.1256144000.2868510721.170894>,
    ncols: 2,
    nrows: 1
  }

  iex(3)> {:ok, tensor} = Tensorflex.int32_tensor vals,dims
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_int32,
    tensor: #Reference<0.1256144000.2868510721.171357>
  }}
  
  ```
  """

  def int32_tensor(%Matrix{nrows: val_rows, ncols: val_cols, data: val_ref}, %Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = NIFs.int32_tensor(val_ref, dim_ref)
    {:ok, %Tensor{datatype: :tf_int32, tensor: ref}} 
  end

  @doc """
  Creates a `TF_INT32` constant value one-dimensional tensor from the integer
value specified.
  
  Takes in an integer value as input.
  
  Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal
Tensorflex struct type that is used for holding tensor data and type.

  ## Examples

  ```elixir
  iex(1)> {:ok, tensor} = Tensorflex.int32_tensor 123
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_int32,
    tensor: #Reference<0.1927663658.3415343105.162588>
  }}
  ```
  
  _Incorrect usage will `raise`_:

  ```elixir
  iex(2)> {:ok, tensor} = Tensorflex.int32_tensor 123.123
  ** (FunctionClauseError) no function clause matching in Tensorflex.int32_tensor/1 

  iex(2)> {:ok, tensor} = Tensorflex.int32_tensor "123.123"
  ** (FunctionClauseError) no function clause matching in Tensorflex.int32_tensor/1  

  ```
  """
  
  def int32_tensor(intval) when is_integer(intval) do
    {:ok, ref} = NIFs.int32_tensor(intval)
    {:ok, %Tensor{datatype: :tf_int32, tensor: ref}}
  end

  @doc """
  Creates a `TF_STRING` constant value string tensor from the string value
specified.
  
  Takes in a string value as input.
  
  Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal
Tensorflex struct type that is used for holding tensor data and type.

  ## Examples

  ```elixir
  iex(1)> {:ok, tensor} = Tensorflex.string_tensor "123.123"
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_string,
    tensor: #Reference<0.2069282048.194904065.41126>
  }}
  
  ```
  
  _Incorrect usage will `raise`_:

  ```elixir
  iex(2)> {:ok, tensor} = Tensorflex.string_tensor 123.123  
  ** (FunctionClauseError) no function clause matching in Tensorflex.string_tensor/1 
  
  iex(2)> {:ok, tensor} = Tensorflex.string_tensor 123    
  ** (FunctionClauseError) no function clause matching in Tensorflex.string_tensor/1
  ```
  """
  
  def string_tensor(stringval) when is_binary(stringval) do
    {:ok, ref} = NIFs.string_tensor(stringval)
    {:ok, %Tensor{datatype: :tf_string, tensor: ref}}
  end

  @doc """
  Allocates a `TF_INT32` tensor of specified dimensions.

  This function is generally used to allocate output tensors that do not hold
any value data yet, but _will_ after the session is run for Inference. Output
tensors of the required dimensions are allocated and then passed to the
`run_session/5` function to hold the output values generated as predictions.

  Takes a Tensorflex `%Matrix` struct matrix as input.

  Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal
Tensorflex struct type that is used for holding the potential tensor data and
type.

  ## Examples
  
  As an example, we can allocate an `int32` output tensor that will be a vector
of 250 values (`1x250` matrix). Therefore, after the session is run, the output
will be an `integer` vector containing 250 values:
  
  ```elixir
  iex(1)> {:ok, tensor} = Tensorflex.create_matrix(1,2,[[1,250]]) |> Tensorflex.int32_tensor_alloc
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_int32,
    tensor: #Reference<0.961157994.2087059457.18950>
  }}
  
  ```
  """

  def int32_tensor_alloc(%Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = NIFs.int32_tensor_alloc(dim_ref)
    {:ok, %Tensor{datatype: :tf_int32, tensor: ref}} 
  end

  @doc """
  Allocates a `TF_FLOAT` tensor of specified dimensions.
  
  This function is generally used to allocate output tensors that do not hold
any value data yet, but _will_ after the session is run for Inference. Output
tensors of the required dimensions are allocated and then passed to the
`run_session/5` function to hold the output values generated as predictions.

  Takes a Tensorflex `%Matrix` struct matrix as input.

  Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal
Tensorflex struct type that is used for holding the potential tensor data and
type.

  ## Examples
  
  As an example, we can allocate a `float32` output tensor that will be a
vector of 250 values (`1x250` matrix). Therefore, after the session is run, the
output will be a `float` vector containing 250 values:
  
  ```elixir
  iex(1)> {:ok, tensor} = Tensorflex.create_matrix(1,2,[[1,250]]) |> Tensorflex.float32_tensor_alloc
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_float,
    tensor: #Reference<0.961157994.2087059457.19014>
  }}
  
  ```
  """
  
  def float32_tensor_alloc(%Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = NIFs.float32_tensor_alloc(dim_ref)
    {:ok, %Tensor{datatype: :tf_float, tensor: ref}} 
  end

  @doc """
  Allocates a `TF_DOUBLE` tensor of specified dimensions.
  
    This function is generally used to allocate output tensors that do not hold
any value data yet, but _will_ after the session is run for Inference. Output
tensors of the required dimensions are allocated and then passed to the
`run_session/5` function to hold the output values generated as predictions.
    
    Takes a Tensorflex `%Matrix` struct matrix as input.
    
    Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal
Tensorflex struct type that is used for holding the potential tensor data and
type.

  ## Examples
  
  As an example, we can allocate a `float64` output tensor that will be a
vector of 250 values (`1x250` matrix). Therefore, after the session is run, the
output will be a `double` vector containing 250 values:
  
  ```elixir
  iex(1)> {:ok, tensor} = Tensorflex.create_matrix(1,2,[[1,250]]) |> Tensorflex.float64_tensor_alloc
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_double,
    tensor: #Reference<0.961157994.2087059457.19025>
  }}
  
  ```
  """

  def float64_tensor_alloc(%Matrix{nrows: dim_rows, ncols: dim_cols, data: dim_ref}) do
    {:ok, ref} = NIFs.float64_tensor_alloc(dim_ref)
    {:ok, %Tensor{datatype: :tf_double, tensor: ref}} 
  end

  @doc """
  Used to get the datatype of a created tensor.

  Takes in a `%Tensor` struct tensor as input.

  Returns a tuple `{:ok, datatype}` where `datatype` is an atom representing
the list of Tensorflow `TF_DataType` tensor datatypes. Click
[here](https://github.com/anshuman23/tensorflex/blob/master/c_src/c_api.h#L98-L122)
to view a list of all possible datatypes.
  
  ## Examples
  
  ```elixir
  iex(1)> {:ok, tensor} = Tensorflex.string_tensor "example"
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_string,
    tensor: #Reference<0.4132928949.2894987267.194583>
  }}
  
  iex(2)> Tensorflex.tensor_datatype tensor
  {:ok, :tf_string}
  ```
  """

  def tensor_datatype(%Tensor{datatype: datatype, tensor: ref}) do
    {:ok, datatype}
  end

  @doc """
  Loads `JPEG` images into Tensorflex directly as a `TF_UINT8` tensor of
dimensions `image height x image width x number of color channels`.

  This function is very useful if you wish to do image classification using
Convolutional Neural Networks, or other Deep Learning Models. One of the most
widely adopted and robust image classification models is the
[Inception](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz)
model by Google. It makes classifications on images from over a 1000 classes
with highly accurate results. The `load_image_as_tensor/1` function is an
essential component for the prediction pipeline of the Inception model (and for
other similar image classification models) to work in Tensorflex.

  Reads in the path to a `JPEG` image file (`.jpg` or `.jpeg`).

  Returns a tuple `{:ok, %Tensor}` where `%Tensor` represents an internal
Tensorflex struct type that is used for holding the tensor data and type. Here
the created Tensor is a `uint8` tensor (`TF_UINT8`).

  __NOTE__: For now, only 3 channel RGB `JPEG` color images can be passed as
arguments. Support for grayscale images and other image formats such as `PNG`
will be added in the future. 

## Examples

  To exemplify the working of the `load_image_as_tensor/1` function we will
cover the entire prediction pipeline for the Inception model. However, this
makes use of many other Tensorflex functions such as `run_session/5` and the
other tensor functions so it would be advisable to go through them first. Also,
the Inception model can be downloaded
[here](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz).
We will be making use of the `cropped_panda.jpg` image file that comes along
with the model to test out the model in Tensorflex.

  First the graph is loaded:

  ```elixir
  iex(1)> {:ok, graph} = Tensorflex.read_graph("classify_image_graph_def.pb")
  2018-07-25 14:20:29.079139: W tensorflow/core/framework/op_def_util.cc:346] Op BatchNormWithGlobalNormalization is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
  {:ok,
  %Tensorflex.Graph{
    def: #Reference<0.542869014.389152771.105680>,
    name: "classify_image_graph_def.pb"
  }}
  ``` 
  Then we load the image as a `uint8` tensor:
  
  ```elixir
  iex(2)> {:ok, input_tensor} = Tensorflex.load_image_as_tensor("cropped_panda.jpg")
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_uint8,
    tensor: #Reference<0.1203951739.122552322.52747>
  }}
  ```
  Then we create the output tensor which will hold out output vector values.
For the Inception model, the output is received as a `1008x1 float32` tensor,
as there are 1008 classes in the model:
  
  ```elixir
  iex(3)> {:ok, output_tensor} = Tensorflex.create_matrix(1,2,[[1008,1]]) |> Tensorflex.float32_tensor_alloc
  {:ok,
  %Tensorflex.Tensor{
    datatype: :tf_float,
    tensor: #Reference<0.1203951739.122552322.52794>
  }}
  ```
  Next, we obtain the results by running the session:

  ```elixir
  iex(4)> results = Tensorflex.run_session(graph, input_tensor, output_tensor, "DecodeJpeg", "softmax")
  2018-07-25 14:33:40.992813: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
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
  Finally, we need to find which class has the maximum probability and identify
it's label. Since `results` is a List of Lists, it's better to read in the
flattened list. Then we need to find the index of the element in the new list
which as the maximum value. Therefore: 
```elixir
  iex(5)> max_prob = List.flatten(results) |> Enum.max
  0.8849328756332397
  
  iex(6)> Enum.find_index(results |> List.flatten, fn(x) -> x == max_prob end)
  169
  ```
  We can thus see that the class with the maximum probability predicted
(`0.8849328756332397`) for the image is `169`. We will now find what the `169`
label corresponds to. For this we can look back into the unzipped Inception
folder, where there is a file called
`imagenet_2012_challenge_label_map_proto.pbtxt`. On opening this file, we can
find the string class identifier for the `169` class index. This is `n02510455`
and is present on Line 1556 in the file. Finally, we need to match this string
identifier to a set of identification labels by referring to the file
`imagenet_synset_to_human_label_map.txt` file. Here we can see that
corresponding to the string class `n02510455` the human labels are `giant
panda, panda, panda bear, coon bear, Ailuropoda melanoleuca` (Line 3691 in the
file). Thus, we have correctly identified the animal in the image as a panda
using Tensorflex.  
"""

  def load_image_as_tensor(imagepath) do
    unless File.exists?(imagepath) do
      raise ArgumentError, "image file does not exist"
    end

    unless (Path.extname(imagepath) == ".jpg" or Path.extname(imagepath) == ".jpeg") do
      raise ArgumentError, "file is not a JPEG image file"
    end
    
    {:ok, ref} = NIFs.load_image_as_tensor(imagepath)
    {:ok, %Tensor{datatype: :tf_uint8, tensor: ref}}
  end

  @doc """
  Loads high-dimensional data from a `CSV` file as a Tensorflex 2-D matrix in a
super-fast manner.

  The `load_csv_as_matrix/2` function is very fast-- when compared with the
Python based `pandas` library for data science and analysis' function
`read_csv` on the `test.csv` file from MNIST Kaggle data
([source](https://www.kaggle.com/c/digit-recognizer/data)), the following
execution times were obtained:
  - `read_csv`: `2.549233` seconds
  - `load_csv_as_matrix/2`: `1.711494` seconds

  This function takes in 2 arguments: a path to a valid CSV file (`filepath`)
and other optional arguments `opts`. These include whether or not a header
needs to be discarded in the CSV, and what the delimiter type is. These are
specified by passing in an atom `:true` or `:false` to the `header:` key, and
setting a string value for the `delimiter:` key. By default, the header is
considered to be present (`:true`) and the delimiter is set to `,`.

  Returns a `%Matrix` Tensorflex struct type.

  ## Examples:
  We first exemplify the working with the `test.csv` file which belongs to the
MNIST Kaggle CSV data
([source](https://www.kaggle.com/c/digit-recognizer/data)), which contains
`28000` rows and `784` columns (without the header). It is comma delimited and
also contains a header. From the `test.csv` file, we also create a custom file
withou the header present which we refer to as `test_without_header.csv` in the
examples below:

  ```elixir
  iex(1)> mat = Tensorflex.load_csv_as_matrix("test.csv")
  %Tensorflex.Matrix{
    data: #Reference<0.4024686574.590479361.258459>,
    ncols: 784,
    nrows: 28000
  }
  
  iex(2)> Tensorflex.matrix_pos mat, 5,97
  80.0
  
  iex(3)> Tensorflex.matrix_pos mat, 5,96
  13.0
  ```
  
  On a visual inspection of the very large `test.csv` file, one can see that
the values in these particular positions are correct. Now we show usage for the
same file but without header, `test_without_header.csv`: 
```elixir
  iex(1)> no_header = Tensorflex.load_csv_as_matrix("test/test_without_header.csv", header: :false)    
  %Tensorflex.Matrix{
    data: #Reference<0.4024686574.590479364.257078>,
    ncols: 784,
    nrows: 28000
  }
  
  iex(2)> Tensorflex.matrix_pos no_header,5,97
  80.0

  iex(3)> Tensorflex.matrix_pos no_header,5,96
  13.0
  ```

  Next we see the delimiter functionalities. First, assuming we have two simple
`CSV` files, `sample1.csv` and `sample2.csv`

  _sample1.csv_:

  ```elixir
  1,2,3,4,5
  6,7,8,9,10
  11,12,13,14,15
  ```

  _sample2.csv_:

  ```elixir
  col1-col2-col3-col4
  1-2-3-4
  5-6-7-8
  9-10-11-12
  ```

  The examples are as follows:
  ```elixir
  iex(1)> m1 = Tensorflex.load_csv_as_matrix("sample1.csv", header: :false) 
  %Tensorflex.Matrix{
    data: #Reference<0.3878093040.3013214209.247502>,
    ncols: 5,
    nrows: 3
  }
  
  iex(2)> Tensorflex.matrix_to_lists m1
  [
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [6.0, 7.0, 8.0, 9.0, 10.0],
    [11.0, 12.0, 13.0, 14.0, 15.0]
  ]

  iex(3)> m2 = Tensorflex.load_csv_as_matrix("sample2.csv", header: :true, delimiter: "-")
  %Tensorflex.Matrix{
    data: #Reference<0.4024686574.590479361.258952>,
    ncols: 4,
    nrows: 3
  }

  iex(4)> Tensorflex.matrix_to_lists m2
  [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
  ```

  _Incorrect usage will `raise`_:
  ```elixir
  iex(1)> not_working = Tensorflex.load_csv_as_matrix("test.csv", header: :no_header, delimiter: ",")
  ** (ArgumentError) header indicator atom must be either :true or :false 
  (tensorflex) lib/tensorflex.ex:122: Tensorflex.load_csv_as_matrix/2
  ```
  """

  def load_csv_as_matrix(filepath, opts \\ []) do
    unless File.exists?(filepath) do
      raise ArgumentError, "csv file does not exist"
    end

    unless (Path.extname(filepath) == ".csv") do
      raise ArgumentError, "file is not a CSV file"
    end

    defaults = [header: :true, delimiter: ","]
    opts = Keyword.merge(defaults, opts) |> Enum.into(%{})
    %{header: header, delimiter: delimiter} = opts
    
    if(header != :true and header != :false) do
      raise ArgumentError, "header indicator atom must be either :true or :false"
    end

    ref = NIFs.load_csv_as_matrix(filepath, header, delimiter)
    {nrows, ncols} = NIFs.size_of_matrix(ref)
    %Matrix{nrows: nrows, ncols: ncols, data: ref}
  end

  @doc """
  Runs a Tensorflow session to generate predictions for a given graph, input
data, and required input/output operations.

  This function is the final step of the Inference (prediction) pipeline and
generates output for a given set of input data, a pre-trained graph model, and
the specified input and output operations of the graph.

  Takes in five arguments: a pre-trained Tensorflow graph `.pb` model read in
from the `read_graph/1` function (`graph`), an input tensor with the dimensions
and data required for the input operation of the graph to run (`tensor1`), an
output tensor allocated with the right dimensions (`tensor2`), the name of the
input operation of the graph that needs where the input data is fed
(`input_opname`), and the output operation name in the graph where the outputs
are obtained (`output_opname`). The input tensor is generally created from the
matrices manually or using the `load_csv_as_matrix/2` function, and then passed
through to one of the tensor creation functions. For image classification the
`load_image_as_tensor/1` can also be used to create the input tensor from an
image. The output tensor is created using the tensor allocation functions
(generally containing `alloc` at the end of the function name).  

  Returns a List of Lists (similar to the `matrix_to_lists/1` function)
containing the generated predictions as per the output tensor dimensions.

## Examples
  
  - A blog post [here](http://www.anshumanc.ml/gsoc/2018/06/14/gsoc/) covers
    generating predictions and running sessions using an MLP model on the Iris
Dataset

  - Generating predictions from the Inception model by Google is covered in the
    `load_image_as_tensor/1` function examples.

  - Working with an RNN-LSTM example for sentiment analysis is covered
    [here](https://github.com/anshuman23/tensorflex/pull/25).  
"""

  def run_session(%Graph{def: graphdef, name: filepath}, %Tensor{datatype: input_datatype, tensor: input_ref}, %Tensor{datatype: output_datatype, tensor: output_ref}, input_opname, output_opname) do
    NIFs.run_session(graphdef, input_ref, output_ref, input_opname, output_opname)
  end
  
end
