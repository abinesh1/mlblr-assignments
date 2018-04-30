# EIP - Assignment 2

## Assignment 2A - Python basics notebook

Github link: https://github.com/abinesh1/mlblr-assignments

## Assignment 2B - Creating a backprop table

#### Input and target variables


``` py
import numpy as np

X = np.array([[1,0,1,0], [1,0,1,1], [0,1,0,1]])
y = np.array([[1],[1],[0]])
```

| x    |      |      |      |
| ---- | ---- | ---- | ---- |
| 1    | 0    | 1    | 1    |
| 1    | 0    | 1    | 1    |
| 0    | 1    | 0    | 1    |

| y    |
| ---- |
| 1    |
| 1    |
| 0    |

#### Step 1: Initializing  weights and biases with random values
```py
wh=np.random.rand(4,3)
bh=np.random.rand(1,3)
wout=np.random.rand(3,1)
bout=np.random.rand(1,1)
```

| wh                                             |      |      |
| ---------------------------------------------- | ---- | ---- |
| 0.05056171 |0.49995133 |-0.99590893 |
| 0.69359851 |-0.41830152 |1.58457724 |
| -0.64770677 |0.59857517 |0.33225003 |
| -1.14747663 |0.61866969|0.08798693  |

| bh   |      |      |
| ---- | ---- | ---- |
|      0.4250724  | 0.33225315 |-1.15681626|

| wout |
| ---- |
|  0.73338017|
|0.40844386|
|0.52790882 |

| bout       |
| ---------- |
| 0.93757158 |

#### Step 2: CAlculate hidden layer input
```python
hidden_layer_input = np.dot(X, wh) + bh
```
| hidden_layer_input |      |      |
| ---- | ---- | ---- |
|    -0.17207266 | 1.43077965 | -1.82047516  |
|   -1.3195493  | 2.04944934 | -1.90846209   |
|     -0.02880573 | 0.53262131 | -2.82938042 |

#### Step 3: Calculate hidden layer activations using sigmoid
```python
sigmoid = lambda x : 1 / (1 + np.exp(-x))
hiddenlayer_activations = sigmoid(hidden_layer_input)
```
|   hidden_input_activations   |      |      |
| ---- | ---- | ---- |
|   -0.17207266 | 1.43077965  | -1.82047516   |
|   -1.3195493  |  2.04944934 | -1.90846209   |
|   -0.02880573 | 0.53262131  |-2.82938042  |
#### Step 4: Linear and non-linear transformation on hidden layer activations
```python
output_layer_input = np.dot(hiddenlayer_activations, wout) + bout
output = sigmoid(output_layer_input)
```
| output     |
| ---------- |
| 0.84237313 |
| 0.82087031 |
| 0.83002062 |
#### Step 5: Calculate error
```python
E = y - output
```
| Error       |
| ----------- |
| 0.15762687  |
| 0.17912969  |
| -0.83002062 |
#### Step 6: Calculate slope at output and hidden layer
```python
slope = lambda x : (sigmoid(x) * (1 - sigmoid(x)))
slope_output_layer = slope(output)
slope_hidden_layer = slope(hiddenlayer_activations)
```
| slope_output_layer |
| ------------------ |
| 0.21041301         |
| 0.21220045         |
| 0.21144303         |

| slope_hidden_layer |      |      |
| ------------------ | ---- | ---- |
|   0.23738353 |0.21333744 |0.2487898  |
|   0.24724073 |0.20671865 |0.24896035 |
|   0.23541567 |0.22674044 |0.2498058  |
#### Step 7: Calculate delta at output layer
```python
learning_rate = 1.0
d_output = E * slope_output_layer * learning_rate
```
| d_output    |
| ----------- |
| 0.03316675  |
| 0.0380114   |
| -0.17550207 |
#### Step 8: Calculate error at hidden layer
```python
error_at_hidden_layer = np.dot(d_output, wout.T)
```
| error_at-hidden_layer |             |             |
| --------------------- | ----------- | ----------- |
| 0.02432383            | 0.01354675  | 0.01750902  |
| 0.02787681            | 0.01552552  | 0.02006655  |
| -0.12870974           | -0.07168274 | -0.09264909 |
#### Step 9: Calculate delta at hidden layer
```python
d_hiddenlayer = error_at_hidden_layer * slope_hidden_layer
```
| d_hidden_layer |             |             |
| -------------- | ----------- | ----------- |
| 0.00577408     | 0.00289003  | 0.00435606  |
| 0.00689228     | 0.00320942  | 0.00499578  |
| -0.03030029    | -0.01625338 | -0.02314428 |
#### Step 10: Updating weights at hidden and output layers
```python
wh = wh + np.dot(X.T, d_hiddenlayer) * learning_rate
wout = wout + np.dot(hiddenlayer_activations.T, d_output) * learning_rate
```
| wh          |            |             |
| ----------- | ---------- | ----------- |
| 0.06322807  | 0.50605078 | -0.98655709 |
| 0.66329822  | -0.4345549 | -1.60772152 |
| -0.63504041 | 0.60467462 | 0.34160187  |
| -1.17088464 | 0.60562573 | -0.10613543 |

| wout       |
| ---------- |
| 0.67006937 |
| 0.35830132 |
| 0.52765534 |
#### Step 11: Updating bias at hidden and output layers
```python
bh = bh + np.sum(d_hiddenlayer, axis=0) * learning_rate
bout = bout + np.sum(d_output, axis=0) * learning_rate
```
| bh         |            |            |
| ---------- | ---------- | ---------- |
| 0.40743847 | 0.32209921 | -1.1706087 |

| bout       |
| ---------- |
| 0.83324766 |