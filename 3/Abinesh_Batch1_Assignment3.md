# **EIP - MLBLR**

## Assignment 3 - Descriptive writeup

## 1. Data Augmentation - Keras

Data augmentation in a deep learning model is used to provide additional training  data to the model by making a small variation to the training data. The variations can include steps such as rotating a specific angle to the left or right cropping a portion of the image, flipping an image horizontally or vertically, varying the pixel density by a small margin etc.. It is a preprocessing step done before training the model with the data. 

![cat](https://m2dsupsdlclass.github.io/lectures-labs/slides/04_conv_nets/images/augmented-cat.png)

#### Example from keras documentation on augmentation

```pytho
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(x_train)

# fits the model on batches with real-time data augmentation:
model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=epochs)

# here's a more "manual" example
for e in range(epochs):
    print('Epoch', e)
    batches = 0
    for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=32):
        model.fit(x_batch, y_batch)
        batches += 1
        if batches >= len(x_train) / 32:
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
```

## 2. Initialization

Weights initialization plays an important role in the performance of the model. Weights can be initialized either as all zeros  or all ones or with random values. Initialization with random weights have proven to be more efficient than useing all zeros or all ones.

### Random initialization

Random initialization of weights generates a random matrix within the range specified. Since the weights are either greater than 0 or 1 and are varied, they can learn different features at each stage of the network rather than each layer of the weights learning the same. In tensorflow weights can be initialized in random by using the following lines of code.

```python
weights = tf.Variable(tf.random_normal([784, 200], stddev=0.35),
                      name="weights")
```

### Xavier initialization

The idea in Xavier initialization is to randomize the initial weights, so that the inputs of each activation function fall within the sweet range of the activation function. Ideally, none of the neurons should start with a trapped situation. This is the same as random initialization but it considers activation function while generating weights.

```py
tf.contrib.layers.xavier_initializer(uniform=True,seed=None, dtype=tf.float32)
```

## 3. Optimizers

Optimization algorithms help to minimize the error function. There are a number of optimization algorithms such as

#### Gradient Descent Optimizer

It is the most common optimizer being used in most deep leaning models. Gradient descent or stochastic gradient descent uses the famous backpropagation to arrive at a more generalized weights for the model. It arrives at the global minima by updating the weights at each step of the training. 

![sgd](https://cdn-images-1.medium.com/max/1600/1*iR7vgbLQ6f70cHHIsSYN2g.png)

The algorithm is as follows:

```python
for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        hidden_input = np.dot(x,weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        output = sigmoid(np.dot(hidden_output,weights_hidden_output))

        ## Backward pass ##
        error = y-output

        output_error_term = error*output*(1-output)

        ## propagate errors to hidden layer

        hidden_error = np.dot(output_error_term,weights_hidden_output)

        hidden_error_term = hidden_error*hidden_output*(1-hidden_output)

        del_w_hidden_output += output_error_term*hidden_output
        del_w_input_hidden += hidden_error_term*x[:, None]

    weights_input_hidden += learnrate*del_w_input_hidden/n_records
    weights_hidden_output += learnrate*del_w_hidden_output/n_records
```

#### Each optimizer is different from the other in a way that they try to avoid local minima. A comparison of all optimizers during training is depicted below.

![opt](http://2.bp.blogspot.com/-q6l20Vs4P_w/VPmIC7sEhnI/AAAAAAAACC4/g3UOUX2r_yA/s1600/s25RsOr%2B-%2BImgur.gif)





