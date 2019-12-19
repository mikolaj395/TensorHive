# TensorFlow MNIST

This example shows distributed training of a simple TensorFlow Keras model using the MNIST dataset.
Two approaches to distributed trainig are presented:
* multiple GPUs on a single node
* multiple nodes (each with potentially multiple GPUs)

Both examples use **tf.distribute.Strategy** module, which is a part of TensorFlow version **2.0** and above.

## Single GPU
Basic implementation, which will be used as a base for distributed examples.
Code: **mnist.py**

## Multiple GPUs on a single node
**mnist.py**
It is actually very easy to run TensorFlow code on multiple GPUs. 
All we need to do is to specify an appropriate strategy:
```
strategy = tf.distribute.MirroredStrategy()
```
And than define and compile our model within the strategy scope:
```
with strategy.scope():
    model = build_and_compile_cnn_model()
```
For further information on **MirroredStrategy** please check the official TensorFlow distributed training [tutorial](https://www.tensorflow.org/guide/distributed_training).
Code: **mnist.py**
## Multiple nodes
todo