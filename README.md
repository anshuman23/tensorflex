# TensorflEx

- The first commit refers to the rather simplistic hello world program for the C API and not in general for Tensorflow. This is present [here](https://www.tensorflow.org/install/install_c#validate_your_installation). This Hello world test currently works!

- The next step is to be able to program a real "Hello World!" TF program. This would involve declaring a string as a Tensorflow constant and then running a session to print the Tensor. As Python is still the best client for TF, this would look something like this in Python:

```python
import tensorflow as tf

with tf.Session() as sess:
    print(sess.run(tf.constant("Hello World!")))
```
- This is going to involve the incorporation of a lot of core functionality (Sessions, Tensors, etc) into TensorflEx from Tensorflow
