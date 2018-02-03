# TensorflEx
### Pre-application Milestones
- [x] __Simple `TF_Version` Hello World test__
    - The first commit refers to the rather simplistic hello world program for the C API and not in general for Tensorflow. This is present [here](https://www.tensorflow.org/install/install_c#validate_your_installation). This Hello world test currently works!
    
- [ ] __Graph, Tensor & Session based proper TF Hello World test__
    - The better hello world test is to be able to program a real "Hello World!" TF program. This would involve declaring a string as a Tensorflow constant and then run a session to print the Tensor in a graph. As Python is still the best client for TF, this would look something like this in Python:

    ```python
    import tensorflow as tf

    with tf.Session() as sess:
        print(sess.run(tf.constant("Hello World!")))
    ```
    
    ```
    OUTPUT: "Hello World!"
    ```
    
    This would require the creation of C bindings capable of creating a new graph, constant string tensors, new operations, creating a session and then running it. This ~~work is currently in progress~~ __now works__. 
    
### How to run these

- Before doing anything, on the command line type `make` (You will see some warnings-- ignore them, they have been taken care of) to compile the C NIF code in `src`.

- __Simple `TF_Version` Hello World test__:
    - Next, open up `iex`:
        ```elixir
        
        iex(1)> c "TensorflEx.ex"
        [TensorflEx]
        
        iex(2)> IO.puts "Hello World! Tensorflow #{TensorflEx.tf_version}"
        Hello World! Tensorflow 1.4.1
        :ok
        ```
        
- __Graph, Tensor and Session based proper TF Hello World test__:
    - Open up `iex`:
        ```elixir
        
        iex(1)> c "TensorflEx.ex"
        [TensorflEx]
        
        iex(2)> graph = TensorflEx.tf_new_graph
        2018-02-03 21:06:07.923328: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
        #Reference<0.85340593.4211212290.58261>
        
        iex(3)> op_desc = TensorflEx.tf_new_op(graph, "Const", "test")
        #Reference<0.85340593.4211212290.58539>
        
        iex(4)> tensor = TensorflEx.tf_string_constant('Hello World!')
        'Hello World!'
        
        iex(5)> IO.puts TensorflEx.tf_create_and_run_sess(graph, op_desc, tensor)
        => [INFO] Loaded Graph correctly
        => [INFO] Loaded Operation Description correctly
        => [INFO] Session Run Complete
        Hello World!
        :ok
        
        
### Summer and post-application milestones
- Will be added with time
