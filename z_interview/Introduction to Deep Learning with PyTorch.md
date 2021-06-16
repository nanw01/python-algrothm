# Introduction to Deep Learning with PyTorch

## Introduction to PyTorch

**Got It!**

## 1. Introduction to PyTorch

Hi, I am Ismail Elezi and I will introduce you to Deep Learning with PyTorch. In this course, we are going to learn the basic concepts of deep learning, a subfamily of machine learning algorithms which has been at the forefront of recent developments in Artificial Intelligence.

## 2. Deep Learning

You might have heard of successes of image classification, machine translation, autonomous driving, Alpha Go or computer bots defeating professional players in Starcraft (AlphaStar). All of these technologies have been empowered by neural networks, another name for deep learning.

## 3. Neural networks

**The magic of neural networks compared to traditional models is in the fact that classical models used one algorithm for feature extraction followed by a machine learning classifier, while neural networks do the optimization altogether.** **The first few layers transform the input into features which are easy to be classified, while the final layer separates the data based on the features which the previous layers have generated.** During the course, we are going to implement many of deep learning central algorithms using the PyTorch library. While the concepts we are going to study are general, the examples we are going to use are mostly computer vision oriented, as are the datasets.

## 4. Why PyTorch?

There are tons of great deep learning libraries right there. We chose PyTorch because of its **simplicity**, because it has **strong GPU support** and it has **already implemented many deep learning algorithms**. It having **strong OOP support** makes a natural choice for many companies like Facebook and Salesforce, while also being one of the most used deep learning libraries in academical research. **Calculating derivatives and gradients is a very important aspect of deep learning algorithms.** Luckily PyTorch is very good at doing it for us. Finally, the library is very **similar to NumPy**, making the switch from NumPy to PyTorch as painless as possible.

## 5. Matrix Multiplication

Matrices are very important in neural networks. The weights and the values of nets are stored in matrices, and many of the operations are done in terms of matrix multiplications. We quickly review hot to multiply two matrices. In order to get the first value (58), we multiply the first row of the first matrix, with the first column of the second matrix, adding in the end each value. Similarly, you do for the other entries.

## 6. PyTorch compared to NumPy

PyTorch's equivalent of NumPy ndarrays is called a torch tensor. You can imagine a tensor being an array with an arbitrary number of dimensions. A tensor can be created by calling torch.tensor(), as you can see in the code block on the left. Like in NumPy, you can create random matrices, by using torch.rand(dim1, dim2). Let's create a random matrix with sizes 2 by 2. Similarly, you can set variables to matrices, and you can check their shape by using .shape function.

## 7. Matrix operations

Multiplying matrices is one of the most common things to do in deep learning. In every neural network you are going to train, there will be millions of matrix multiplications. PyTorch supports matrix multiplication via the **torch.matmul()** function, as can be seen in the example here.

## 8. Matrix operations

Another important operator is element-wise multiplication (where every element in the first matrix is multiplied by the corresponding element in the second matrix), which can be performed in PyTorch via the **asterisk (*) operator.**

## 9. Zeros and Ones

Some special types of matrices are matrices of zeros, matrices of ones and identity matrices. These matrices in PyTorch can be created by using torch.zeros(), torch.ones(), and torch.eye() functions; very similar to numpy.zeros(), numpy.ones() and numpy.identity().

## 10. PyTorch to NumPy and vice versa

It is easy to convert NumPy arrays to torch tensors, it can be done via function from_numpy(). Similarly, you can convert torch tensors to NumPy arrays via the numpy() function.

## 11. Summary

We have prepared a summary of matrix operations, so don't hesitate to get back to it if you forget the names of the functions.

## 12. Let's practice

Let us practice with the concepts we just learned!





```python
# Import torch
import torch

# Create random tensor of size 3 by 3
your_first_tensor = torch.rand((3, 3))

# Calculate the shape of the tensor
tensor_size = your_first_tensor.shape

# Print the values of the tensor and its shape
print(your_first_tensor)
print(tensor_size)

# Create a matrix of ones with shape 3 by 3
tensor_of_ones = torch.ones(3, 3)

# Create an identity matrix with shape 3 by 3
identity_tensor = torch.eye(3)

# Do a matrix multiplication of tensor_of_ones with identity_tensor
matrices_multiplied = torch.matmul(tensor_of_ones, identity_tensor)
print(matrices_multiplied)

# Do an element-wise multiplication of tensor_of_ones with identity_tensor
element_multiplication = tensor_of_ones * identity_tensor
print(element_multiplication)
```



# Forward propagation

**Got It!**

## 1. Forward propagation

Welcome back. Here we are going to explain forward propagation (also known as forward pass), an important step on training and evaluation of all classifiers.

## 2. Why forward propagation?

Let's take a classifier, maybe a neural network. No worries if you don't know what it is, I promise that you are going to soon learn about it, and the principle is general for most classifiers. The first step is to evaluate what classifier is doing given your data. Given some data (the yellow layer), the model does all the operation in the following (blue) layers until it gives some result in the output (brown) layer. This step is called the forward step.

## 3. Forward pass of a simple computational graph

We are going to start with a simple example. Let's assume that we have a model, it could be a neural network, some other type of classifier, or in more abstract terms a computational graph. Intuitively, a computational graph is a network of nodes that represent numbers, scalars, or tensors and are connected via edges that represent functions or operations. For simplicity and visualization, instead of using large multi-dimensional tensors, we are going to use scalars. Our graph has nodes a, b, c and d and some operations between the nodes, like adding node a with node b, or multiplying node c with node d. Our job is to do all the operations until we get the result g.

## 4. Forward pass of a simple computational graph

First we add a and b, getting 2 + (-4) = -2, putting the result in node e. Similarly, we multiply c with d, getting (-2) * 2 = -4, and we put the result in node f. Now our nodes in the graph are nodes e and f, containing values (-2) and (-4). The last step is to multiply these two values and to put the result in g.

## 5. Forward pass of a simple computational graph

We do that, getting (-2) * (-4) = 8, and so finishing our exercise.

## 6. PyTorch implementation

Let's implement it on PyTorch. We first initialize tensors a, b, c and d, to their corresponding values 2, (-4), (-2) and 2. We then add a and b to tensor e, and we multiply c with d, putting the result in tensor f. Finally, we multiply e with f, putting the result in tensor g, and then we print the values of these three tensors, getting the same result as in the slides. Now, there is hardly anything fancy in this exercise, however understanding it is of utter importance. Neural networks (and most of the other classifiers) can be understood as computational graphs (in fact, your code gets converted to a computational graph), just that the graphs will be much larger and the tensors within them will contain millions of entries. An additional benefit of computational graphs, is that they make the automatic computation of derivatives (or gradients) much easier, as we will see in the next lesson.

```python
import torch
a = torch.Tensor([2])
b = torch.Tensor([-4])
c = torch.Tensor([-2])
d = torch.Tensor([2])

e = a + b
f = c * d

g = e * f
print(e, f, g)

# tensor([-2.]), tensor([-4.]), tensor([8.])
```

## 7. Let's practice!

Let's do a quick exercise to solidify the knowledge about computational graphs and forward pass.

```python
# Initialize tensors x, y and z
x = torch.rand(1000, 1000)
y = torch.rand(1000, 1000)
z = torch.rand(1000, 1000)

# Multiply x with y
q = x * y

# Multiply elementwise z with q
f = z * q

mean_f = torch.mean(f)
print(mean_f)
```





# Backpropagation by auto-differentiation

## 1. Backpropagation by auto-differentiation

In this lesson, we are going to introduce the main algorithm in neural networks, the so-called backpropagation algorithm, and see how we can use it on PyTorch. This lesson is a bit more theoretical than most of the lessons in the course, but there is no need to get scared of that.

## 2. Derivatives

Derivatives are one of the central concepts in calculus. In layman's terms, the derivatives represent the rate of change in a function, so where the function is rapidly changing, the absolute value of derivatives is high, while when the function is not changing, the derivatives are close to 0. They could also be interpreted as describing the steepness of a function. For example, in the function here, points A and C have large derivatives, the line is steep in these positions, while point B has a very small derivative. If you haven't ever heard about derivatives, I would highly recommend taking a look at them, on Khan Academy for example.

## 3. Derivative Rules

Some important rules of derivatives are the addition and multiplication rule. The addition (or sum) rule says that for two functions f and g, the derivative of their sum is the sum of their individual derivatives. On the other hand, the multiplication rule says that the derivative of their product, is f times derivative of g plus g times derivative of f. Derivative of a number times a function, is the number, for example, the derivative of 3x is 3. The derivative of a number itself is always 0. The derivative of something with respect to itself is always 1. Another important rule is chain rule which deals with the composition of functions. In the example in the slide, the derivative of f(g(x)) is derivative of f under function g(x) times derivative of g(x). A closely related term with derivatives is the gradient. The gradient is a multi-variable generalization of the derivative, and considering that neural networks have many variables, we will typically use the term gradient instead of derivative when we will be working with neural networks.

## 4. Derivative Example - Forward Pass

Let's make things concrete with an example. We have 3 variables, x, y and z having values -3, 5 and -2. First we add x and y in variable q, which is -3 + 5 = 2. Then we multiply z with q, giving us -4.

## 5. Derivative Example - Backward Pass

Now, let's compute the derivatives. We put them in the blue boxes below the values of the nodes. The derivative of f with respect to itself is 1.

## 6. Derivative Example - Backward Pass

The derivative of q is z (because of multiplication rule) times the derivative of f (chain rule), giving us -2 times 1 equals to -2.

## 7. Derivative Example - Backward Pass

Similarly, the derivative of z is q times derivative of f, equals to 2.

## 8. Derivative Example - Backward Pass

Finally, we compute derivatives of x and y. Because of the sum rule, their derivative is 1 times derivative of q (from chain rule) giving us -2. The logic of calculations is the same regardless if we have 5 variables, or 1 billion variables.

## 9. Backpropagation in PyTorch

The derivatives are calculated in PyTorch using the reverse mode of auto-differentiation, so you will rarely need to write code to calculate derivatives. Let's compute the derivatives of the example in the previous slide. First, we initialize tensors x, y and z with values -3, 5 and -2. However, we need to set requires_grad flag to True, in order to tell PyTorch that we need their derivatives. Then we add x and y in q, and we multiply q with z in f. Finally, we write f.backward() to tell PyTorch to compute the derivatives. Results are the same as when we calculated them by hand. `tensor.grad` simply gets the gradient of that tensor.

## 10. Let's practice

Let's practice some derivatives.



```python

# Initialize x, y and z to values 4, -3 and 5
x = torch.tensor(4., requires_grad=True)
y = torch.tensor(-3., requires_grad=True)
z = torch.tensor(5., requires_grad=True)

# Set q to sum of x and y, set f to product of q with z
q = x + y
f = q * z

# Compute the derivatives
f.backward()

# Print the gradients
print("Gradient of x is: " + str(x.grad))
print("Gradient of y is: " + str(y.grad))
print("Gradient of z is: " + str(z.grad))


```



```python

# Multiply tensors x and y
q = x * y

# Elementwise multiply tensors z with q
f = z * q

mean_f = torch.mean(f)

# Calculate the gradients
mean_f.backward()

```



# Introduction to Neural Networks

## 1. Introduction to Neural Networks

In this lecture, we are going to introduce neural networks. More precisely, we are going to study fully connected neural networks, the simplest form of modern neural networks.

## 2. Other classifiers

In Supervised Learning with scikit-learn, you have seen some classifiers like **k-nn**. There are many other classifiers, some of which are very good, like **Random Forests**, Adaboost or **Support Vector Machines**. These classifiers work well when the data is given on vectorial format, as features.

## 3. ANN vs other classifiers

However, most of the data is not given as features. Instead, the data is in some rich format, like images, speech, text or video. In those cases, what people did before, was to use another algorithm to extract those features. In computer vision, the majority of last decade's research was on finding algorithms which get good features from images. Perhaps the most famous of those algorithms was the SIFT algorithm, which given an image, returns features from that image. Then those features are classified using a classifier like SVM. Maybe you can see the problem here. In order to solve the problem, we are optimizing two different algorithms (SIFT and SVM) which aren't related at all with each other. Neural networks work a bit differently. They have an input layer (in the figure denoted with 1), one or more hidden layers (denoted with 2), and an output layer (denoted with 3). The job of the hidden layers is to get good features, while the job of the output layer is to classify those features. With the net being trained end-to-end, we have a single algorithm which at the same time both finds good features, and classifies them. This has shown to work very well, and so neural networks have revolutionized many fields, to the point of making the old algorithms obsolete.

## 4. Fully connected neural networks

Let's have a closer look at the network given here. Each neuron (or better called unit to avoid analogy with biological neurons) in a layer is connected with every unit in both the previous and the following layers. These connections are called weights, and are represented by a matrix (tensor). As input, we have `input_layer` containing 10 units. In order to get the values of the first hidden layer `h1`, we multiply the vector of features with the first matrix of weights, called `w1`. Look at the matrix of weights, the first dimension should always correspond to the preceding layer, while the second dimension to the following layer. As you can imagine, `h1` contains 20 units. Similarly, we continue for the second hidden layer `h2`, which is the product of the first hidden layer `h1` and the second matrix of weights `w2`. Finally, we get the results of the output_layer, which has 4 classes, by multiplying the second hidden layer `h2` with the third matrix of weights `w3`.

## 5. Building a neural network - PyTorch style

Writing neural networks this way is a bit complicated. Fortunately, PyTorch has a better way of doing it, which is object-oriented. We define a class, let's call it Net, which inherits from nn.Module (look that we are importing torch.nn). In the __init__ method, we define our parameters, the tensors of weights. For fully connected layers, they are called nn.Linear. The first parameter is the number of units of the current layer, while the second parameter is the number of units in the next layer. In the forward method, we apply all those weights to our input. Finally, we instantiate our model, by calling class Net, and we get the result, by applying object net over our input_layer.

```python

import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 20)
        self.output = nn.Linear(20, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

input_layer = torch.rand(10)
net = Net()
result = net(input_layer)
```



## 6. Let's practice!

Let's practice building some neural networks.



```python

# Initialize the weights of the neural network
weight_1 = torch.rand(784, 200)
weight_2 = torch.rand(200, 10)

# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)

# Multiply hidden_1 with weight_2
output_layer = torch.matmul(hidden_1, weight_2)
print(output_layer)

```



```python

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Instantiate all 2 linear layers  
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200,10)

    def forward(self, x):
      
        # Use the instantiated layers and return x
        x = self.fc1(x)
        x = self.fc2(x)
        return x

```





# Artificial Neural Networks

# Activation functions

**Got It!**

## 1. Activation functions

In the last lecture we talked about neural networks. We saw how they can be implemented in PyTorch, it looked pretty simple.

## 2. Motivation

Let's look at the neural network drawn here. We see that it has one input layer with 2 units, containing numbers 2 and 1, one hidden layer and one output layer. We can implement this neural network in PyTorch as we did before, with the code given here, where first we multiply input layer with weight_1, then we multiply their product with weight_2, and in the end we print the results of the output layer, getting two numbers 0.9696 and 0.7527.

## 3. Matrix multiplication is a linear transformation

Now, we will try to do something different. Let's first multiply the matrices of weights as we do here with **torch.matmul()**, and then we'll multiply the input with the product of these two matrices. If we print the results, we see something interesting. The result of the output layer is exactly the same as before. This means that we can achieve the exact result by using a single layer neural network, with this particular set of weights. You may be wondering, is this unexpected? Not really. Linear algebra demonstrates that matrix multiplication is actually a linear transformation, meaning that we can simplify any neural network in a single layer neural network.

## 4. Non linearly separable datasets

This comes with an irritating consequence, our neural nets are not that powerful. Using them makes possible to only separate linearly separable datasets, like the one in the left. These nets can separate only datasets that in 2D space are separable by a straight line, in 3D space by a plane, and in a higher dimensional space by a hyperplane. Unfortunately, most datasets actually are not linearly separable, they are messy and look more like the dataset in the right. We need to somehow capture the non-linear relationships between the data, and so we need to build much more powerful models to deal with these datasets.

## 5. Activation functions

Because of this, we use something which is called an activation function. Activation functions are non-linear functions which are inserted in each layer of the neural network, making neural networks nonlinear and allowing them to deal with highly non-linear datasets, thus making them much more powerful.

![image-20210322195912095](https://i.loli.net/2021/03/23/APTcyjUVvf1X7Wb.png)

## 6. ReLU activation function

The most used activation function in neural networks is called ReLU standing for Rectified Linear Unit. Its graph can be seen here and its analytical formula is given by the equation in the right. As you can see, this function accepts a tensor in input and it sets all of its entries that are negative to 0, while it doesn't change the positive entries. Despite being such a simple function, neural networks which use ReLU are extremely powerful, and some of these neural networks are used in complex problems like image recognition, autonomous driving or Alpha Go. ReLU is already implemented in PyTorch - as you might expect - and is part of the torch-dot-nn submodule. It can be initialized as an object of nn-dot-ReLU class. We can see its behavior by looking at the code in the left-lower part, and the results of that code in the other part of the slide. The positive inputs are effectively conserved, while negative inputs are now set to 0.

## 7. Let us implement some activation functions

In PyTorch besides ReLU there are many other activation functions which are already implemented. Let's start using them and see how they work.



```python

# Apply non-linearity on hidden_1 and hidden_2
hidden_1_activated = relu(torch.matmul(input_layer, weight_1))
hidden_2_activated = relu(torch.matmul(hidden_1_activated, weight_2))
print(torch.matmul(hidden_2_activated, weight_3))

# Apply non-linearity in the product of first two weights. 
weight_composed_1_activated = relu(torch.matmul(weight_1, weight_2))

# Multiply `weight_composed_1_activated` with `weight_3
weight = torch.matmul(weight_composed_1_activated, weight_3)

# Multiply input_layer with weight
print(torch.matmul(input_layer, weight))

```



```python

# Instantiate ReLU activation function as relu
relu = nn.ReLU()

# Initialize weight_1 and weight_2 with random numbers
weight_1 = torch.rand(4, 6)
weight_2 = torch.rand(6, 2)

# Multiply input_layer with weight_1
hidden_1 = torch.matmul(input_layer, weight_1)

# Apply ReLU activation function over hidden_1 and multiply with weight_2
hidden_1_activated = relu(hidden_1)
print(torch.matmul(hidden_1_activated, weight_2))

```



# Loss functions

**Got It!**

## 1. Loss functions

Now that we know how to build neural networks, it is time to start learning how to train them. So far, all neural networks in this course had random weights, so they weren't particularly useful.

## 2. Loss Functions

The recipe for training neural networks is the following. We first initialize the weights with random numbers, and then do a forward pass. Next step is to calculate the loss function, which gives us a single number. Finally, we calculate the gradients using back propagation, and in the end we change the weights based on the computed gradients. We will go back to this in more detail in lesson 2.4. Here, we are going to concentrate on this loss function. The easiest way to understand loss function (also called cost function), is for it to be a measure on how wrong the model is performing. In the prerequisite course about supervised learning, you have seen the loss of linear regression, the least squared error, measuring the squared distances between predictions and ground truth. For classification, which is the topic of this lesson, we use a loss function called softmax cross-entropy. Loss functions are an important topic in deep learning, and there is much research going on it. For complicated problems, we have complicated loss functions, containing multiple terms.

![image-20210322200931567](https://i.loli.net/2021/03/23/Gg3ACfJKkvt4aUQ.png)

## 3. Softmax Cross-Entropy Loss

One important thing about loss functions is that they should be differentiable, otherwise we wouldn't be able to compute its gradients. For this reason, instead of using accuracy (which is not differentiable), we need to use some proxy loss functions, and in neural nets, a softmax function followed by cross entropy performs really well. Softmax is a function which transforms numbers into probabilities. Let's do an example. We want to classify this image here, and the neural network has given us these scores: 3.2 for cat, 5.1 for car, and -1.7 for frog.

$$\sigma_{i}(z)=\frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}}$$

## 4. Softmax Cross-Entropy Loss

First thing is the numerator in the formula. We exponentiate each term (setting e to the power of each score), giving us unnormalized probabilities, numbers which don't sum to 1.

## 5. Softmax Cross-Entropy Loss

Then in the denominator, where each term is divided by the sum of all terms, for example the sum of the unnormalized probabilities is 188.68 so 24.5/188.68 gives us 0.13. After we do this step, we get probabilities. In this case the net is predicting the image to be 13% cat, 87% car, and 0% frog.

## 6. Softmax Cross-Entropy Loss

Finally, we compute the cross-entropy loss. It is just the minus logarithm (with base e) of the probability of the correct class. You can calculate yourself that -ln(0.13) is 2.0404. Things to note: if probability of the correct class is 1, then loss is 0, the net is working perfectly. If the probability is close to 0, then the loss is big, possibly infinite, and the predictions are way off.

## 7. CE loss in PyTorch

Let's code the example in PyTorch. We have the tensor of predicted scores (called logits) with the same values as in the example, and the ground truth is the first class (cat). For loss function, typically initialized with the word criterion we choose CrossEntropyLoss() which actually combines softmax with cross-entropy. We calculate the loss by simply writing criterion(logits, ground_truth) where the logits are the scores for each class, and we get the same result as in the example.

## 8. CE loss in PyTorch

Now, let's see what would have happened if cat class had a high prediction score, 10.2. We see that the loss is so small, only 0.0061. In contrast, if the cat got a prediction score of -10, the loss would have been quite big, 15.1. Rule of thumb, the more accurate the network is, the smaller the loss is.

## 9. Let's practice!

Let's practice with loss functions!





What is the result of the **softmax cross-entropy** loss function?

| Class | Predicted Score |
| :---- | :-------------- |
| Cat   | -1.2            |
| Car   | 0.12            |
| Frog  | 4.8             |



$$\sigma_{i}(z)=\frac{e^{z_{i}}}{\sum_{j=1}^{K} e^{z_{j}}}$$

![image-20210322202121793](https://i.loli.net/2021/03/23/OsJW1YCQkcfiwTp.png)



```python

# Initialize the scores and ground truth
logits = torch.tensor([[-1.2,0.12,4.8]])
ground_truth = torch.tensor([2])

# Instantiate cross entropy loss
criterion = nn.CrossEntropyLoss()

# Compute and print the loss
loss = criterion(logits,ground_truth)
print(loss)

```



```python

# Import torch and torch.nn
import torch
import torch.nn as nn

# Initialize logits and ground truth
logits = torch.rand(1,1000)
ground_truth = torch.tensor([111])

# Instantiate cross-entropy loss
criterion = nn.CrossEntropyLoss()

# Calculate and print the loss
loss = criterion(logits, ground_truth)
print(loss)

```









# Preparing a dataset in PyTorch



## 1. Preparing a dataset in PyTorch

Welcome back! In this lesson, we are going to shift gears a bit, and instead of talking about neural networks, we are going to talk about datasets, and how we can use them in PyTorch.

## 2. MNIST and CIFAR-10

During the course, we are going to use 2 datasets: MNIST and CIFAR-10. MNIST is arguably the most famous dataset in computer vision. It is a dataset containing 70000 handwritten digits, with each class containing 6000 images for training, and 1000 for testing. The images have shape (28, 28) and are in grayscale format (black and white), so they have one channel. CIFAR-10 on the other hand is a dataset of natural images. It contains 50000 images for training, and 10000 for testing, with each class having in total 6000 images. Images have shape (32, 32) and they are given in RGB format, so they have 3 channels (1 each for red, green and blue). Both datasets are part of PyTorch, and during the lectures we are going to use the CIFAR-10 dataset, while the exercises will be in MNIST dataset.

## 3. Datasets and Dataloaders

In order to be able to use datasets in PyTorch, they need to be in some PyTorch friendly format, a format that the framework will be able to understand. Let's see how we can achieve it with CIFAR-10 dataset. First thing, we need to do some imports that we haven't seen so far. In addition to torch, we import also torchvision (a package which deals with datasets and pretrained neural nets) and torch.utils.data. From torchvision submodule we import torchvision.transforms. After it, we define a transformation of images to torch tensors, using transforms.ToTensor() function. While we are here, we also define the values for standardizing the images, the mean and standard deviation of each channel. We have already computed the mean and std, so we just use the precomputed numbers.

## 4. Datasets and Dataloaders

Next thing is to get the dataset. CIFAR-10 comes as part of torchvision.datasets. We decide where is (or will be) the dataset, in this case in ./data folder. For training set we set flag train to true, while for testing set we set it to False. We also set the download flag to True, which tells the PyTorch that if dataset is not in the specified folder, to download and put it there. Finally, we set transform to transform, essentially transforming images to torch tensors by applying the transformation we defined in the previous slide. We build trainloader and testloader, getting the data ready for PyTorch. A DataLoader object comes as part of torch.utils.data module. The first argument of the object is the dataset. Then we decide the size of the minibatch. Our dataset is too large to be used entirely, instead we decide for each iteration to use only 32 randomly sampled images. Random part comes from shuffle flag. And finally, we decide how many processes we are going to use to fetch the data in num_workers.

## 5. Inspecting the dataloader

It is possible to inspect the dataloader. For example, we can look at the shape of the testing dataset, or the shape of the training dataset. Similarly, we can look at the minibatch size. It might also be interesting to look at the type of the random sampler, in this case being an object of class torch.utils.data.sampler.RandomSampler.

## 6. Let's practice!

Your turn now! Let's practice with MNIST dataset.

```python

# Transform the data to torch tensors and normalize it 
transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize((0.1307), ((0.3081)))])

# Prepare training set and testing set
trainset = torchvision.datasets.MNIST('mnist', train=True, 
									  download=True, transform=transform)
testset = torchvision.datasets.MNIST('mnist', train=False,
			   download=True, transform=transform)

# Prepare training loader and testing loader
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
										 shuffle=False, num_workers=0)     

```



```python
# Compute the shape of the training set and testing set
trainset_shape = trainloader.dataset.train_data.shape
testset_shape = testloader.dataset.test_data.shape

# Print the computed shapes
print(trainset_shape, testset_shape)

# Compute the size of the minibatch for training set and testing set
trainset_batchsize = trainloader.batch_size
testset_batchsize = testloader.batch_size

# Print sizes of the minibatch
print(trainset_batchsize, testset_batchsize)
```







# Training neural networks

**Got It!**

## 1. Training neural networks

Welcome back! Now that we know how to build neural networks, and we also know how to use datasets in PyTorch, it is time to combine everything we learned so far and start training neural networks!

## 2. Recipe for training neural networks

Remember the recipe we saw in the Loss Functions lesson? Here we are going to go in more depth on how to apply that recipe in order to train neural networks in PyTorch. Initially we create the dataloaders for the dataset we want the neural network to train on. Then, it is time to build a neural network. By default, all parameters of a neural network are initialized with random numbers. If you want you can use different strategies on the matter of initialization, but nowadays it is not important so you can just trust PyTorch to do it for you. After that, you do a forward pass (using a minibatch) at the end of which you compute the loss function, giving you a number which tries to measure how good the neural network is working in the training set. Then we compute the gradients using backpropagation, and finally we change the weights (parameters) of the neural network, using a variation of the stochastic gradient descent algorithm (SGD). SGD simply applies the formula in the right: from each weight it subtracts the gradient of the weight times a learning rate. We will explain later how to find good learning rates, and its importance. For now, you can consider it as a constant which is not too big (because then it will overshoot the minima), and not too small (because in that case the training will be very slow). We iterate over the last 4 steps many times.

![image-20210322205427726](https://i.loli.net/2021/03/23/cGXkSgaIKLHplAu.png)

## 3. Gradient descent

What does changing the weights based on gradients mean? If you remember this slide from the last chapter, you know that gradients can be interpreted as the steepness of a function. By going in the direction of the gradient (subtracting the gradient of weights from weights) we are actually going in direction of the local minima of a function. And the entire training of the network is based on minimizing the loss function. This algorithm is called gradient descent.

## 4. Recap - Dataloaders

We saw in the last lesson how to create dataloaders. In this example we are going to use CIFAR-10 dataset. Nothing special in this slide, the same information as in the last lesson. After all this code, we have prepared CIFAR-10 dataset to be used in PyTorch.

## 5. Neural Networks - Recap

Then we write the class for the neural network. Considering that CIFAR-10 has images of shape (32, 32, 3) as input layer we have 32 * 32 * 3 units. We decide to have 500 units in the hidden layer, it is a hyperparameter, so it is a decision made by us. With the dataset having 10 classes, we put 10 units in the output layer. You might also see the nn.functional module we imported, which does the same thing as nn module, but in a more functional way. We use it to apply relu nonlinearity in the hidden layer.

```python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 500)
        self.fc2 = nn.Linear(500, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

```



## 6. Training the Neural Network

Here we do the training. First we instantiate the net, **the loss (cross-entropy) and the optimizer (Adam)**. We chose the Adam optimizer which works very well, and is a version of gradient descent. Then we loop 10 times over the entire dataset. We use zero_grad() function in order to not accumulate gradients from the previous iterations. The forward step is done using net(inputs), giving us the result (in this case output). We compute the loss function in the next line, and then we compute the gradients using loss.backward(). Finally, we change the weights using our optimizer with the optimizer.step() command. The line inputs = inputs.view(-1, 32 * 32 * 3) simply puts all the entries of the images into vectors.

```python
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)
for epoch in range(10): # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        # Get the inputs
        inputs, labels = data
        inputs = inputs.view(-1, 32 * 32 * 3)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

## 7. Using the net to get predictions

Finally, trained nets are used to make predictions on unseen images. We first set the net in test (evaluation) mode using net.eval(). The network gives us scores for each class, and we get the class with the highest score (using max function) as prediction. We save the predictions and compute the accuracy.

```python

correct, total = 0, 0
predictions = []
net.eval()
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    inputs = inputs.view(-1, 32*32*3)
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predictions.append(outputs)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))

```



## 8. Let's practice!

Let's practice training neural networks.





```python
# Define the class Net
class Net(nn.Module):
    def __init__(self):    
    	# Define all the parameters of the net
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28 * 1, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):   
    	# Do the forward pass
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```



```python

# Instantiate the Adam optimizer and Cross-Entropy loss function
model = Net()   
optimizer = optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()
  
for batch_idx, data_target in enumerate(train_loader):
    data = data_target[0]
    target = data_target[1]
    data = data.view(-1, 28 * 28)
    optimizer.zero_grad()

    # Complete a forward pass
    output = model(data)

    # Compute the loss, gradients and change the weights
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

```







# Convolution operator

**Got It!**

## 1. Convolution operator

Welcome back to the third chapter of the course.

## 2. Problems with the fully-connected neural networks

We already saw how to build and train neural networks. However, using only fully connected neural networks comes with a few problems. First, do we need to consider the relationship between all the features?

## 3. Problems with the fully-connected neural networks

Consider the red pixels in the left-upper part and the blue pixels in the right-lower part. They are so far away from each other, that their relationship doesn't tell us much for the general picture. Yet fully connected networks take into consideration their relationship. Doing so comes at the cost of efficiency. With so many units and weights, these networks become too large, so training becomes computationally inefficient. Additionally, with so many parameters (weights), they can overfit on the training set.

## 4. Main ideas

The two main ideas on mitigating these problems are that the units should be connected with only a few units from the previous and the next layer, and that the units should share weights. A very famous class of networks that implement these ideas is the convolutional neural networks (CNNs). We will start studying them by first understanding the convolution operator.

## 5. Convolutions - basic idea

The idea of convolutions is relatively simple. Given an image (in this case the red one with shape 32 by 32 by 3) and a kernel or filter (the blue one, with shape 5 by 5 by 3) where the depth dimension must match, all we need to do is to convolve the image in each position with the filter. In analogy with fully connected neural networks, the filters here are actually the weights of the network, and we get good features by using them.

## 6. Convolving

Convolving means doing a dot product between the image and the filter. So, for each position of the image we will get a single number.

## 7. Activation map

After we finish the procedure, we get what is called a feature (or activation map). The size has changed from 32 to 28, because in the last 4 positions of the image we couldn't do a convolution (size of the filter must be smaller than the size of the image).

## 8. Activation map

A convolution layer is a layer that contains multiple activation maps. The goal of CNNs is to actually learn filters which give us good activation maps, good features. We will go back to this soon.

## 9. Padding

Finally, if we want the size of the activation map to match the size of the image, we can do padding, adding 0s in the sides of the image. By doing so, we ensure that every value (number in the image) can be used for the convolutions, and so we don't decrease the size of the feature maps.

## 10. Convolutions in PyTorch

There are two ways of using convolutions in PyTorch. In an object-oriented way as part of torch.nn module or in a functional way as part of torch.nn.functional. In both cases, you need to give to the function the image, the size of the filter (or the filter itself), the stride and the padding parameters. Let's see an example how it can be done in both cases.

## 11. Convolutions in PyTorch

In both cases, we have a minibatch of 16 images, each with shape (3, 32, 32) and we want to do a single convolution, with size of the kernel being 5 by 5, stride 1, and no padding. Note that in torch.nn we need to create a Conv2d filter with these parameters (input channels, output channels, filter size, stride, padding) while in functional, you simply create a random filter. Then you apply the convolution in both cases and print the results, getting the same shape in output. We didn't mention stride yet, it is the distance between spatial locations where the convolution filter is applied.

## 12. Convolutions in PyTorch

If we want to change the number of convolutional filters to 5 like we had in the slides before, we simply need to change the in_channels parameter to 5 (and have 5 as the first dimension of the filter in functional case). Similarly, if we want to add padding, we just put 1 in the padding parameter.

## 13. Let's practice!

Let's practice!



```python

# Create 10 random images of shape (1, 28, 28)
images = torch.rand(10, 1, 28, 28)

# Build 6 conv. filters
conv_filters = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)

# Convolve the image with the filters
output_feature = conv_filters(images)
print(output_feature.shape)

```



```python

# Create 10 random images
image = torch.rand(10, 1, 28, 28)

# Create 6 filters
filters = torch.rand(6, 1, 3, 3)

# Convolve the image with the filters
output_feature = F.conv2d(image, filters, stride=1, padding=1)
print(output_feature.shape)


```





# Pooling operators



## 1. Pooling operators

We saw in the last video the convolution operator, which is the main building block in convolutional neural networks (CNNs). Another very important layer in CNNs is the pooling operator, which can come in two different ways. The convolutions are used to extract features from the image, while the pooling is a way of feature selection, choosing the most dominant features from the image, or combining different features. Additionally, they lower the resolution of the images, making the computations more efficient.

## 2. Pooling layer

Let's assume that we have a feature map, maybe built after applying convolutional filters, which has dimensions for width and height set to 224, and it has depth (number of channels) 64. Pooling is simply lowering the spatial dimension (width and height), typically by 2. This is done for 2 reasons: making computations more efficient (smaller feature maps are easier to deal with), and making the learning invariant to translations. This means making the network robust to shifting (movements) of the image. The two most important pooling operators are max-pooling and average-pooling.

## 3. Max-Pooling

Max-pooling simply takes the maximum number in regions of images. Typically for pooling, we consider filters with size 2 by 2, and we use stride equals to 2. In this example, we first consider the pink part (the first 2 by 2 region) containing numbers 3, 1, 6 and 0, taking its maximum value, number 6. Then we jump 2 places (stride is set to 2) in the green part, taking the number 9 as the maximum. Similarly, we do for the lower part of the image, taking the maximum values 3 and 4. If convolutional layers were learning the features, pooling layers actually do feature selection, selecting the strongest activations in the feature map, and making the learning subject to translation invariance (by considering only the largest values in patches of the image, we make learning invariant to small shifting/translation).

## 4. Average-Pooling

Average-pooling is another important pooling operator which is used typically in deep networks, in the later stages of them. It is very similar to max-pooling: instead of taking the maximum value in a patch of the image, it takes the average value. As you can see it here, for the pink value it takes the mean of the region, first adding numbers 3, 1, 6 and 0 (the sum being 10) and then dividing by 4, finally getting the value 2.5

## 5. Max-pooling in PyTorch

Let's see how we can use pooling (both in the OOP and functional way) in PyTorch. We first initialize our image, in this case being the same image we saw in the previous slides. Multiple brackets are needed, because the image needs to have 4 dimensions (for minibatch size, depth, height and width). We then build the max pooling objects with kernel size 2 and apply it to our image when we use torch.nn(). Or, we directly give the image to the F.max_pool2d() function when we use the functional way. Finally, we print the results.

## 6. Average pooling in PyTorch

In order to do average pooling, we apply the same procedure. We just replace MaxPool2d() with AvgPool2d() if we use torch.nn(), and we replace max_pool2d with avg_pool2d if we use torch.nn.functional().

## 7. Let's practice!

Let's practice some pooling exercises.



```python

# Build a pooling operator with size `2`.
avg_pooling = torch.nn.AvgPool2d(2)

# Apply the pooling operator
output_feature = avg_pooling(im)

# Use pooling operator in the image
output_feature_F = F.avg_pool2d(im,2)

# print the results of both cases
print(output_feature)
print(output_feature_F)

```





# Convolutional Neural Networks

**Got It!**

## 1. Convolutional Neural Networks

Great job with pooling operators! You have learned many concepts during the course: you know how to build fully-connected neural networks, and how to train them using gradient descent-like algorithms and backpropagation. You have also learned about the convolution and pooling operators. Now it is time to go a step further and learn how to build convolutional neural networks (CNNs), the most important technique in deep learning.

## 2. AlexNet

While CNNs have existed for decades, their resurgence happened in 2012, when Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton published the so-called AlexNet paper and smashed every record in image classification. Until that time, people were aware of the existence of CNNs, but they didn't take them seriously. However, after that influential publication, everything changed and the entire field of machine learning, and especially computer vision, got transformed forever.

## 3. Transformation of computer vision

Almost everything in computer vision is empowered by CNNs. If not, they at least play a large part on it. Detection, segmentation, recognition, autonomous driving, AlphaGo, Starcraft zero, you name it, every one of them is either a CNN, or it has a large CNN as an important building block.

## 4. AlexNet architecture

Now, let's go back and see what exactly is a CNN, studying the AlexNet architecture in particular. In essence, it is simply a neural network containing multiple convolutional and pooling layers, potentially followed by one or more fully connected layers. If you look at the scheme here, you can see that the net contains 5 convolutional layers, 3 max pooling and an average pooling layer, and in the end it has 3 fully-connected layers, the last of which classifies images in 1000 different classes.

1. 1 Alex Krizhevsky, Ilya Sutskever and Geoffrey Hinton; ImageNet Classification with Deep Convolutional Neural Networks, NIPS 2012.

## 5. AlexNet in PyTorch

Coding AlexNet in PyTorch is surprisingly easy. Let's see how it can be done. We first build a class called AlexNet which inherits from nn.Module. Then we start writing the __init__ method, where we pass the number of classes as an argument, in this case 1000. We call the superclass using the super operator, and then we start declaring all the parameters we want to have. In particular, we see that we have 5 convolutional layers, from conv_1 to conv_5, each with different number of filters. We got the numbers of filters from the paper. Then we want to have 3 pooling layers. They all have the same kernel_size and stride, so we define it only once. Similarly, we define once the ReLU nonlinearity. Finally, we have the three fully-connected layers, the last of which contains the number of classes.

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
```

## 6. The forward method

Now all that remains is implementing the forward method. We start by passing our image to conv_1() followed by ReLU non-linearity and a maxpooling layer. Then we continue with the second convolutional and pooling layers. Then we have 3 conv layers followed by a max-pooling and an average pooling layer. Finally, we have three fully connected layers. Now the net will return the predicted class for each image we will pass to it. Of course, in order for it to do a correct prediction, it first needs to be trained. Building the net is simply a matter of creating an object from this class.

```python
def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.maxpool(x)
    x = self.relu(self.conv2(x))
    x = self.maxpool(x)
    x = self.relu(self.conv3(x))
    x = self.relu(self.conv4(x))
    x = self.relu(self.conv5(x))
    x = self.maxpool(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.relu(self.fc1(x))
    x = self.relu(self.fc2(x))
    return self.fc3(x)
net = AlexNet()
```



## 7. Let's practice!

Let's practice building some CNNs before we learn on how to train them.



```python
class Net(nn.Module):
    def __init__(self, num_classes):
        super(Net, self).__init__()
		
        # Instantiate the ReLU nonlinearity
        self.relu = nn.ReLU()
        
        # Instantiate two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1)
        
        # Instantiate a max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Instantiate a fully connected layer
        self.fc = nn.Linear(7 * 7 * 10, 10)

    def forward(self, x):

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # Apply conv followd by relu, then in next line pool
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # Prepare the image for the fully connected layer
        x = x.view(-1, 7 * 7 * 10)

        # Apply the fully connected layer and return the result
        return self.fc(x)
```





# Training Convolutional Neural Networks

## Imports

```python
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
```

## Dataloaders

```python
transform = transforms.Compose(
[transforms.ToTensor(),
transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
shuffle=False, num_workers=2)
```

## Building a CNN

```python
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, adding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 4 * 4)
        return self.fc(x)
```

## Optimizer and Loss Function

```python
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=3e-4)
```

## Training a CNN

```python
for epoch in range(10):
    for i, data in enumerate(trainloader, 0):
        # Get the inputs
        inputs, labels = data
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
    print('Finished Training')
```

## Evaluating the results

```python
correct, total = 0, 0
predictions = []
net.eval()
for i, data in enumerate(testloader, 0):
    inputs, labels = data
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predictions.append(outputs)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print('The testing set accuracy of the network is: %d %%' % (100 * correct / total))

```



# The sequential module

**Got It!**

## 1. The sequential module

In this chapter we are going to study some more advanced techniques, which will allow us to train neural networks better and more easily. While the effect of these techniques is small in simple neural networks, making it hard to appreciate them, they are a must when working with big neural networks, and knowing them will make a big difference. We start the chapter by studying the sequential module, which is a PyTorch tool allowing us to build neural networks in an easier way.

## 2. AlexNet - declaring the modules

Remember the AlexNet architecture from the last chapter? We needed to individually declare every layer we wanted to have in the network.

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
```



## 3. AlexNet - forward() methods

And after that, we needed to apply each of those layers (sequentially). While for simple architectures like AlexNet this isn't a big deal, imagine doing this for modern architectures containing hundreds of layers, some of them being very complicated. It's a lot of code to write.

```python
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
net = AlexNet()
```



## 4. The sequential module - declaring the modules

Fortunately, in PyTorch there exists an easier way to abstract this, allowing us to create the same neural networks while using much less code. It is called 'Sequential Module' and is very useful for feedforward networks (where the flow goes in one direction). By using this module, you can divide your network in parts which logically make sense, like convolutional layers, or fully connected layers. You can also reuse the modules to create similar blocks in the neural network. Let's see how this can be done in PyTorch. We define all the convolutions, poolings, fully-connected layers etc same as before, but now the order of operators matters also in declaration. Additionally, we encapsulate them within nn.Sequential(). In this case we are using one sequential module for the feature extraction part (convolutions and poolings), and one for the classification part (fully connected layers). This is a very optimized OOP way of doing things, and allows you to change parts of the network independently from each other.)



```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(), 
            nn.Linear(256 * 6 * 6, 4096), 
            nn.ReLU(inplace=True),
            nn.Dropout(), 
            nn.Linear(4096, 4096), 
            nn.ReLU(inplace=True), 
            nn.Linear(4096, num_classes)
        )
```



## 5. The sequential module - forward() method

You remember how long the forward() method was before?! Now, by using the sequential module, instead of applying each operation, we actually need to apply each sequential module. Initially, we give the images to the first module which we called features and contains all the convolutional and pooling layers. Then we do an average pooling and finally we apply the classifier containing three fully connected layers. The network is exactly the same as before, it is just much simpler to code and debug. When using larger and more complicated networks, using the sequential module lets you write tens of lines of code vs hundreds of them.

```python
def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), 256 * 6 * 6)
    x = self.classifier(x)
    return x
```



## 6. Let's practice!

Let's practice using sequential module!





```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2), nn.ReLU(inplace=True)
        )
        
        # Declare all the layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 40, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 2048), 
            nn.ReLU(inplace=True),
            nn.Linear(2048, 10)
        )
        
    def forward(self, x):
      
        # Apply the feature extractor in the input
        x = self.features(x)
        
        # Squeeze the three spatial dimensions in one
        x = x.view(-1, 7 * 7 * 40)
        
        # Classify the images
        x = self.classifier(x)
        return x
```





# The problem of overfitting

**Got It!**

## 1. The problem of overfitting

Welcome back! Now, that you know how to train fully-connected neural networks and convolutional neural networks, we are going to see how we can help the training procedure and make it better.

## 2. Overfitting

So far, we have talked only about what happens within the training set. In reality, we are mostly interested by the generalization performance of the network; in other words, how good the network performs in classifying examples which haven't been part of the training set. Arguably the biggest problem in machine and deep learning is overfitting, where the model (the neural network) works very well on the training set, but its performance is much worse on the testing set. This typically happens when we have a very complicated and non-smooth hypothesis (or separator), like the green one. Instead, we want simple and smooth separators like the black one.

## 3. Detecting overfitting

You can detect overfitting by plotting the accuracy of your algorithm (and this stands not only for neural networks, but for most algorithms) in both the training and testing set. In the example in the slide, we have plotted the accuracy of the model on the training set with the blue line, and the accuracy of the model on the testing set with the green line. As you can see, there is a large error on the testing set, and a large gap between both lines (represented by the red arrow). When the difference in accuracy between training and testing is large, we have a case of overfitting, also called high variance.

## 4. Detecting overfitting

On the other hand, if the gap in accuracy is small, then we have a case of low-variance, which means that no overfitting is happening, and the results in the testing set are a good predictor of the general performance of the algorithm.

## 5. Overfitting in the testing set

We will spend the next lesson on ways how to prevent overfitting. For now though, we are going to see how not to make a mistake that many people do (and I did it deliberately in the last two slides). If you train multiple networks on the same dataset (by changing the hyperparameters like learning rate or number of layers) and then check the accuracy of each of them on both the training and testing sets, you might eventually find one network which performs well on the testing set. While this may seems satisfying, the problem is that if the same testing set is used many times, then it is contaminated (making the results of it not trustworthy, and optimistic), and instead of telling you how good the model is, it actually tells you how good the model is on this particular test set. To prevent this, what wise people do is to actually introduce the validation set (also known as cross validation).

## 6. Validation set

Now, **the networks are trained in the training set as before, and each of them is tested in the validation set.** Finally, **the best performing model is tested in the testing set**, and that number is reported as the accuracy of the model. It is very important that the **testing set is used only once** (or at most, a few times), otherwise **its results won't be trustworthy**. Additionally, **it is very important that the training, validation and testing sets do not overlap with each other.**

## 7. Using validation sets in PyTorch

Creating validation sets in PyTorch is easy. All you need to do is to split the training set into two non-overlapping sets as we did in the code in the slide, by using the indices variable. In these examples, we used the first 45000 data points for training and the remaining 5000 for validation. We achieve this by using SubsetRandomSampler, which randomly divides one dataset into multiple ones. The remaining procedure is the same, we just replace testing set with the validation set for evaluation, and finally, we do final testing in the testing set with the best performing model.

## 8. Let's practice!

Let's practice some exercises with the validation set.

```python
indices = np.arange(50000)
np.random.shuffle(indices)
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), 
                    (0.229, 0.224, 0.225)
                )
            ]
        )
    ),
    batch_size=1, 
    shuffle=False, 
    sampler=torch.utils.data.SubsetRandomSampler(
        indices[:45000]
    )
)
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), 
                    (0.229, 0.224, 0.225)
                )
            ]
        )
    ),
    batch_size=1, 
    shuffle=False,
    sampler=torch.utils.data.SubsetRandomSampler(
        indices[45000:50000]
    )
)
```











```python
# Shuffle the indices
indices = np.arange(60000)
np.random.shuffle(indices)

# Build the train loader
train_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', download=True, train=True,
                     transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                     batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[:55000]))

# Build the validation loader
val_loader = torch.utils.data.DataLoader(datasets.MNIST('mnist', download=True, train=True,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])),
                   batch_size=64, shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices[-5000:]))
```







# Regularization techniques

**Got It!**

## 1. Regularization techniques

In this lesson, we are going to introduce a few ways of preventing overfitting. Like I mentioned at the beginning of the chapter, it is very hard to see the effect of these techniques in small neural networks. However, more or less every modern large network uses some of these (if not all) techniques, allowing those models to be trained more efficiently and to give better predictions.

## 2. L2-regularization

The first method we are going to introduce is **l2-regularization**, a method also used in algorithms like regression or SVM. To apply l2 regularization, we just add the second term in the loss function. As you can see from the second sum, this term penalizes large weights, which affects training in a bad way. To use l2 regularization in PyTorch, all you need to do is to add the weight_decay argument in the optimizer.

## 3. Dropout

**Dropout is another regularization technique used in neural networks.** During each forward pass, there is a probability p for each unit to be dropped from the computation. By doing so, units are forced to not be dependent on the surrounding units. It can also be considered that in each iteration, the architecture of the neural network is changing (because different neurons are removed from computations with each iteration). Typically, dropout is used in fully-connected layers, while is rarely used in convolutional layers.

1. 1 Srivastava, Hinton, Krizhevsky, Sutskever, Salakhutdinov; Dropout: a simple way to prevent neural networks from overfitting, JMLR 2014

## 4. Dropout in AlexNet - PyTorch code

We can see how dropout can be used in the AlexNet example. We simply add nn.Dropout() in the fully connected layers, with parameter p controlling the probability of units being dropped. In this case, when we use p=0.5 we are essentially asking PyTorch to drop each unit with probability 50%.

```python
self.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(inplace=True),
    nn.Linear(4096, num_classes),
)
```



## 5. Batch-normalization

**Batch normalization is a very important technique used nowadays in practically every neural network.** In layman's terms, **it computes the mean and the variance of the minibatch for each feature, and then it normalizes the features based on those stats.** While the effect can't be seen in small neural networks, nowadays it's unthinkable to train large neural networks without using batch-normalization. Like other techniques, it can be easily used in PyTorch, using the BatchNorm2d function, where num_features is the number of features, and the other two arguments are hyperparameters.

1. 1 Ioffe and Szegedy; Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, ICML 2015

```python
self.bn = nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.9)
```

## 6. Early-stopping

The last technique we will present is early stopping. It simply checks the accuracy of the network in the validation set at the end of each epoch, and if after n epochs the performance of the net hasn't increased (or it has decreased), then training is terminated. It is important to save the network every time the accuracy improves, so that at the end of the training we use the best performing network.

## 7. Hyperparameters

How do we choose all the hyperparameters for the methods presented in the previous few slides? Usually, we train many neural networks (tens or even hundreds of them), with different hyperparameters, typically set at random. We measure the accuracy of each network in the validation set, and finally test the best performing one in the testing set to report the accuracy of our model. Remember, it is important to not run more more than a few (or even better, just one) tests on the testing set.

![image-20210323132436746](https://i.loli.net/2021/03/24/oQ9bkLAzO3x6tPi.png)

## 8. Eval() mode

Some of the techniques mentioned here (like dropout and batch-norm) behave differently when the net is getting trained and when the net is getting evaluated. We need to manually tell PyTorch if you are training or evaluating the net. We do this by using .train() when we are training the net and .eval() when we are evaluating the net. It is very important to set the net in the correct mode, otherwise the training and evaluation will be broken.

## 9. Let's practice!

Let's practice some of the techniques we studied in this section!



```python
# Instantiate the network
model = Net()

# Instantiate the cross-entropy loss
criterion = nn.CrossEntropyLoss()

# Instantiate the Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.001)
```

## Dropout

```python

class Net(nn.Module):
    def __init__(self):
        
        # Define all the parameters of the net
        self.classifier = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 10))
        
    def forward(self, x):
    
    	# Do the forward pass
        return self.classifier(x)

```

# Batch-normalization

```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Implement the sequential module for feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2), 
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(10),
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2), 
            nn.ReLU(inplace=True), 
            nn.BatchNorm2d(20))
        
        # Implement the fully connected layer for classification
        self.fc = nn.Linear(in_features=7*7*20, out_features=10)
```



# Transfer learning

**Got It!**

## 1. Transfer learning

Welcome back! We are going to study one of the most important techniques in **training convolutional neural networks**.

## 2. Features in CNN

An interesting discovery in CNN research was the deeper you progress in the network, the more abstract the features become. The first few layers in a CNN (at the bottom here) detect basic features like edges or corners. When you go a bit deeper in the CNN, you get more complicated features like circles or squares. Eventually you reach more abstract features like wheel or eye, while in the final layers you might get features which detect for example red cars or athletic shoes. A nice consequence of this, is that the low-level features are very general and, to a large degree, dataset independent.

## 3. Transfer Learning

We trained all nets so far from scratch, initializing them with random weights. However, in practice this isn't how things are usually done. Instead of training the net from scratch, we download a net trained on another dataset (typically that is a big dataset like ImageNet containing 1.2 millions of images), and then we retrain the net in our dataset. This allows us not only to achieve significantly better results in less training time, but also to train networks on very small datasets (containing only hundreds of images). Remember how people used to say that neural networks can only be used on large datasets? Forget about it, with this technique, you can train large neural networks on very small datasets. In literature, this 'retraining' is typically called '**finetuning**', but the essence is the same. **There are two ways of finetuning neural networks.** **One is to freeze most of the layers (not updating them during back-propagation) and finetuning only the last few layers (or only the very last one), while the other technique is to finetune everything.** Typically, if your dataset is extremely small, it is a good idea to freeze most of the layers, in order to avoid overfitting.

## 4. Finetuning in PyTorch

Let's say we have trained a net on CIFAR-10, which we have saved as 'cifar10_net.pth'. The net is a large CNN, the penultimate layer of which has shape (4, 4, 1024). So it has spatial dimensions of 4 and 1024 features. Then let's say we have a similar dataset, called CIFAR-100 which contains 100 classes. Furthermore, let's suppose that this dataset is much smaller. Training there from scratch will result in overfitting, but finetuning the net trained in CIFAR-10 will result in very high accuracy. To do this in PyTorch, we first load the old model using load_state_dict(). Then we change the number of units in the last layer: it should always correspond to the number of classes, in this case 100. The training procedure is the same as it would have been if we trained from scratch.

```python
# Instantiate the model
model = Net()
# Load the parameters from the old model
model.load_state_dict(torch.load('cifar10_net.pth'))
# Change the number of out channels
model.fc = nn.Linear(4 * 4 * 1024, 100)
# Train and evaluate the model
model.train()
```



## 5. Freezing the layers

If we want to freeze all the layers, except the classifier, then all we need to do is iterate over the parameters of the model, using model.parameters(). We then set their required_grad flag to False. In this case, only the final layer is updated.

```python
# Instantiate the model
model = Net()
# Load the parameters from the old model
model.load_state_dict(torch.load('cifar10_net.pth'))
# Freeze all the layers bar the final one
for param in model.parameters():
param.requires_grad = False
# Change the number of output units
model.fc = nn.Linear(4 * 4 * 1024, 100)
# Train and evaluate the model
model.train()
```



## 6. Torchvision library

**Torchvision is a PyTorch library with many pretrained networks, ready to be used for your dataset.** For example, you can use state-of-the-art models like resnets, by simply downloading them from torchvision with a single line of code, as shown here. Then you just need to change the number of units to the number of classes in your dataset, and voila, you have a very good net ready to be used, without even needing to write the code for it. In my experience with neural networks, for every network I have trained from scratch, I have finetuned around 100s of networks. So when you'll have problems to solve with a CNN, always consider finetuning an already pretrained net.

```python
import torchvision
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, num_classes)
```



## 7. Let's practice!

Let's practice finetuning some CNNs.





```python

# Create a new model
model = Net()

# Change the number of output units
model.fc = nn.Linear(7 * 7 * 512, 26)

# Train and evaluate the model
model.train()
train_net(model, optimizer, criterion)
print("Accuracy of the net is: " + str(model.eval()))


# Create a model using
model = Net()

# Load the parameters from the old model
model.load_state_dict(torch.load('my_net.pth'))

# Change the number of out channels
model.fc = nn.Linear(7 * 7 * 512, 26)

# Train and evaluate the model
model.train()
train_net(model, optimizer, criterion)
print("Accuracy of the net is: " + str(model.eval()))



```



```python
# Import the module
import torchvision

# Download resnet18
model = torchvision.models.resnet18(pretrained=True)

# Freeze all the layers bar the last one
for param in model.parameters():
    param.requires_grad = False

# Change the number of output units
model.fc = nn.Linear(512, 7)
```

















