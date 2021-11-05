# ml-glossary
Some basic machine/ deep learning terminologies to revisit anytime.

# Deep learning | Machine learning | Neural network

1.  **Loss Function**: A way of measuring how far off predictions are from the desired outcome. The measured different is called ‘loss’.

2. **Weights(parameter)**: A weight represent the strength of the connection between units. If the weight from node 1 to node 2 has greater magnitude, it means that neuron 1 has greater influence over neuron 2. A weight brings down the importance of the input value.

3.  **Bias:** The bias value allows the activation function to be shifted to the left or right, to better fit the data. ... You can think of the bias as a measure of how easy it is to get a node to fire.

4. **Optimizer function**: A way of adjusting internal values in order to reduce the loss. In other words, Optimizers are algorithms or methods used to minimize an error function (loss function) or to maximize the efficiency of production. i.e., GD, SGD, **Adam**(**ADA**ptive with **M**omentum)…

5. **Learning Rate**: The learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated. The “step size” for loss improvement during gradient descent.

6. **Activation function**: an activation function is a function that is added into an artificial neural network in order to help the network learn complex patterns in the data.

   > a.    Linear (A straight line function where activation is proportional to input ( which is the weighted sum from neuron ))

   > b.    ELU (Exponential Linear Unit)

   > c.    ReLU (Rectified Linear Units)

   > d.    LeakyReLU (variant of ReLU)

   > e.    Sigmoid (between 0 and 1)

   > f.    Tanh (number to the range [-1, 1])

   > g.    Softmax (Softmax function calculates the probabilities distribution of the event over ‘n’ different events)

7. **Node:** A deep learning node is "a *computational unit* that has one or more weighted input connections, a transfer function that combines the inputs in some way, and an output connection. Nodes are then organized into layers to comprise a network

8.  **Layer**: A neural network is made up of vertically stacked components called Layers.

   1. a.    Input;

      b.    Hidden;

      c.    Output.

9. **Autoencoder**: Autoencoder is an unsupervised artificial neural network that learns how to efficiently compress and encode data then learns how to reconstruct the data back from the reduced encoded representation to a representation that is as close to the original input as possible. Autoencoder, by design, reduces data dimensions by learning how to ignore the noise in the data.

10. **Backpropagation**: property that allows the weights between nodes to be optimized through a process called Backpropagation, and what allows a network to learn from its mistakes. Backpropagation is so named because the weights are updated backwards, from output towards input. The calculation of internal variable(W&B) adjustments according to the optimizer algorithm, starting from the output layer and working back through each layer to the input.

11. **Batch Normalization**: The idea is to normalize the inputs of each layer in such a way that they have a mean output activation of zero and standard deviation of one. This is analogous to how the inputs to networks are standardized. It is a technique for improving the performance and stability of neural networks.

12.  **Feature:** Inputs| Labels: Outputs

13. **Dense and Fully Connected (FC)**: Each node in one layer is connected to each node in the previous layer.

14. **Epoch:** A full pass over the entire training dataset

15. **Flattening:** The process of converting a 2d/3d image into 1d vector

16. **ReLU:** An activation function that allows a model to solve nonlinear problems.

17. **Softmax:** A function that provides probabilities for each possible output class.

18. **Regression:** A model that outputs a single value. For example, an estimate of a house’s value.

19. **Classification:** A model that outputs a probability distribution across several categories.

20. **CNNs**: Convolutional neural network. That is, a network which has at least one convolutional layer. A typical CNN also includes other types of layers, such as pooling layers and dense layers.

21. **Convolution**: The process of applying a kernel (filter) to an image

22. **Kernel / filter**: A matrix which is smaller than the input, used to transform the input into chunks

23. **Padding:** Adding pixels of some value, usually 0, around the input image

24. **Pooling:** The process of reducing the size of an image through downsampling. There are several types of pooling layers. For example, average pooling converts many values into a single value by taking the average. However, maxpooling is the most common.

25. **Maxpooling:** A pooling process in which many values are converted into a single value by taking the maximum value from among them.

26. **Stride:** the number of pixels to slide the kernel (filter) across the image.

27. **Downsampling**: The act of reducing the size of an image

