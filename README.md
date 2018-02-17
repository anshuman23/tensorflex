# TensorflEx

## Contents
- [x] __How to run__
- [x] __Documentation__ 
- [x] __Pre-application Milestones__
- [ ] __Summer and post-application milestones__

### How to run
- Clone this repository and `cd` into it
- Run `mix deps.get` to install the dependencies
- Run `mix compile` to compile the code
- Open up `iex` using `iex -S mix`

### Documentation
Tensorflow requires the creation of graphs to represent the operations in a network. It also requires the creation of tensors, which are just the basic multi-dimensional datatype that contain data arrays and matrices for computation purposes. Once the graph is created an operation also needs to be created, which contains information about the kind of graph we are going to be running. The graph is finally run in a _session_, in which input and output tensors are defined.

For the purposes of describing the current capabilities of TensorflEx, the same things hold true. After getting `iex` started, we first define a new graph:
```elixir
iex(1)> graph = TensorflEx.new_graph
```
Next, we define the graph operation. Here `"Const"` is the type of operation and `"test"` is the name we have given it so that we can reference it when we want:
```elixir
iex(2)> op_desc = TensorflEx.new_op(graph, "Const", "test")
```
Then we define our string constant tensor. Make sure you do not use double quotes ("") in describing the tensor input as it has to be a char list:
```elixir
iex(3)> tensor = TensorflEx.string_constant('Hello World!')
```
Lastly, we create and simultaneously run the session for our graph and then print the output:
```elixir
iex(4)> IO.puts TensorflEx.create_and_run_sess(graph, op_desc, tensor)
```
This will give us the value of the string constant tensor as output. 

### Pre-application Milestones
- [x] __Simple `TF_Version` Hello World test__
    - The first commit refers to the rather simplistic hello world program for the C API and not in general for Tensorflow. This is present [here](https://www.tensorflow.org/install/install_c#validate_your_installation). This Hello world test works!

    __Code in TensorflEx__:
    
    ```elixir
    
      iex(1)> c "TensorflEx.ex"
      [TensorflEx]
        
      iex(2)> IO.puts "Hello World! Tensorflow #{TensorflEx.version}"
      Hello World! Tensorflow 1.4.1
      :ok
      
    ```
       
- [x] __Graph, Tensor & Session based proper TF Hello World test__
    - The better hello world test is to be able to program a real "Hello World!" TF program. This would involve declaring a string as a Tensorflow constant and then run a session to print the Tensor in a graph. As Python is still the best client for TF, this would look something like this in Python:

    ```python
    import tensorflow as tf

    with tf.Session() as sess:
        print(sess.run(tf.constant("Hello World!")))
    ```
    
    ```
    OUTPUT: "Hello World!"
    ```
    
    __Code in TensorflEx__:
        
    ```elixir
        
        iex(1)> c "TensorflEx.ex"
        [TensorflEx]
        
        iex(2)> graph = TensorflEx.new_graph
        2018-02-03 21:06:07.923328: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
        #Reference<0.85340593.4211212290.58261>
        
        iex(3)> op_desc = TensorflEx.new_op(graph, "Const", "test")
        #Reference<0.85340593.4211212290.58539>
        
        iex(4)> tensor = TensorflEx.string_constant('Hello World!')
        'Hello World!'
        
        iex(5)> IO.puts TensorflEx.create_and_run_sess(graph, op_desc, tensor)
        => [INFO] Loaded Graph correctly
        => [INFO] Loaded Operation Description correctly
        => [INFO] Session Run Complete
        Hello World!
        :ok
   ```     
        
### Summer and post-application milestones
- Will be added in time
