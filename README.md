# TensorflEx
### Pre-GSoC Milestones
- [x] __Simple `TF_Version` Hello World test__
    - The first commit refers to the rather simplistic hello world program for the C API and not in general for Tensorflow. This is present [here](https://www.tensorflow.org/install/install_c#validate_your_installation). This Hello world test currently works!

- [ ] __Graph, Tensor and Session based proper TF Hello World test__
    - The next step is to be able to program a real "Hello World!" TF program. This would involve declaring a string as a Tensorflow constant and then run a session to print the Tensor. As Python is still the best client for TF, this would look something like this in Python:

    ```python
    import tensorflow as tf

    with tf.Session() as sess:
        print(sess.run(tf.constant("Hello World!")))
    ```
    
### How to run these

- __Simple `TF_Version` Hello World test__:
    - On the command line type `make`
    - Next, open up `iex`:
        ```elixir
        
        iex(1)> c "TensorflEx.ex"
        [TensorflEx]
        
        iex(2)> IO.puts "Hello World! Tensorflow #{TensorflEx.tf_version}"
        Hello World! Tensorflow 1.4.1
        :ok
        ```
