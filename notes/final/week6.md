# Artificial Neural Network

## Neural Network Models

Sometimes called "Connectionist Models". Inspired by human network of neurons. For more detail of human neurons:

- Switching time: ~ .001 secs
- Number of neurons: ~ 10<sup>10</sup>
- Connections per neuron: ~ 10<sup>4-5</sup>
- Scene recognition time: ~ .1 secs

While rather slow compared to computers, but with an objectively inferior condition, a human is able to learn to recognise by vision rather quickly, even during juvenile state because of many parallel computation.

From these inspiration, an artificial neural nets (ANN's) is invented with these property:

- Neuron-like threshold switching units
- Many weighted interconnections among units
- Highly parallel
- Emphasis on tuning weight automatically

### When to consider NN?

- Input is high-dimensional, whether discreet or continuous. (e.g. raw sensor input)
- Output is discrete or continuous. Can also be vector!
- Possibly noisy data
- Form of target function is unknown. (unlike other traditional ML where we have to define a target function. As the NN will take care of this.)
- Human readability of result is not important

**Example usage**:

- ASR
- Image Classification
- Q&A, Converastional chatbot

<img width="546" alt="Screenshot 2568-03-24 at 15 32 54" src="https://github.com/user-attachments/assets/46efd8ac-ed53-4f55-bc2d-5a551f5c2c79" />

## Perceptron

<img width="690" alt="Screenshot 2568-03-24 at 15 40 39" src="https://github.com/user-attachments/assets/5d88d9e0-2475-4a72-8e9e-9c11b9414986" />

> This perception implement a **bipolar** activation function that returns either -1 or 1. While there is also **binary** activation function that returns 0 and 1 as well. This depends totally on our task.

```math
o(x_1, \dots , x_n) =
\left\{
\begin{array}{l}
\ 1 \text{ if } w_0 + w_1 x_1 + \cdots + w_n x_n > 0 \\
-1 \text{ otherwise}
\end{array}
\right.
```

### Decision surface of perceptron

For one perceptron, the decision is rather similar to that of Logistic Regression with different activation function.

<img width="597" alt="Screenshot 2568-03-24 at 15 44 01" src="https://github.com/user-attachments/assets/4f37d561-a2ad-4a73-ab3c-fb90c69702c9" />

Perceptrons can be used to represent some decision operator or some useful functions. For example:

- What weights represent

```math
g(x_1 , x_2) = AND(x_1 , x_2)?
```

- Truth value of $AND$:

<img width="128" alt="Screenshot 2568-03-24 at 15 46 52" src="https://github.com/user-attachments/assets/5e3a6105-e592-4740-992c-0c36a7364287" />

- Decision boundary of $AND$:

<img width="176" alt="Screenshot 2568-03-24 at 15 47 15" src="https://github.com/user-attachments/assets/5056767b-f4a2-43c0-b96f-f0e1ffca36c1" />

**But some functions are not representible by a single perceptron**

- e.g., not linearly separable.
  - like $XOR$ function:
    - Truth value of $XOR$:  
      <img width="132" alt="Screenshot 2568-03-24 at 15 50 22" src="https://github.com/user-attachments/assets/79f5dcbe-07cd-4848-baf8-db9fdab36464" />
    - Decision boundary of $XOR$:  
      <img width="238" alt="Screenshot 2568-03-24 at 15 50 40" src="https://github.com/user-attachments/assets/9259f256-3809-4be6-94a8-4831632272c4" />

- We simply use more than a single perceptron!

### Perceptron training rule

```math
w_i \leftarrow w_i + \Delta w_i
```

where

```math
\Delta w_i = \eta (\underbrace{t - o}_{\text{error}})x_i
```

Where:

- $t = c(\vec{x})$ &rarr; target value
- $o$ &rarr; perceptron output
- $\eta$ &rarr; learning rate

This is proven that it will converge, **IF**:
- training data is linearly separable
- and $\eta$ is sufficiently small

problem with this is:
- the formular was not define after optimisation &rarr; will keep running even it is converged.

### Gradient descent

To make it simple as we need to use the **squared error** function, we will now use a **linear** activation function, where

```math
o = w_0 + w_1 x_1 + \cdots + w_n x_n
```
> In other word, this is easier to differentiate.

Squared error:

```math
E[\vec{w}] \equiv \frac{1}{2} \sum_{d \in D} (t_d - o_d)^2
```
> The $\frac{1}{2}$ is there just to make it more elegant after differentiation.

Where $D$ is the training data.

<img width="386" alt="Screenshot 2568-03-24 at 16 12 12" src="https://github.com/user-attachments/assets/3f48851a-3b86-4bd3-bfe2-e5f9645e5c79" />

Gradient:

```math
\nabla E[\vec{w}] \equiv \left[ \frac{\partial E}{\partial w_0} , \frac{\partial E}{\partial w_1} , \cdots , \frac{\partial E}{\partial w_n} \right]
```

Training rule:

```math
\Delta \vec{w} = - \eta E[\vec{w}]
```

i.e.,

```math
\Delta w_i = - \eta \frac{\partial E}{\partial w_i}
```

> because the result will always be positive (i.e., going towards local/global "maximum") after differentiation, the negative sign has to be put in front of the update term to make it go in different way.

after differentiation here we have:

```math
\frac{\partial E}{\partial w_i} = \sum_d (t_d - o_d)(-x_{i , d})
```

therefore,

```math
\Delta w_i = \eta \sum_d (t_d - o_d)x_{i , d}
```

quite similar to $\Delta w_i = \eta (t - o)x_i$, but the key difference is the $o$ and $o_d$ as we use linear activation function for gradient descent, which will make sure that the algorithm will end up converging, while the $o$ in perceptron learning rule is an output after binary or bipolar function, which will keep the algorithm going back and forth, and does not converge!

#### Pseudocode

**GRADIENT-DESCENT (training-examples, $\eta$)**

Each training example is a pair of the form $\langle \vec{x} , t \rangle$, where $\vec{x}$ is the vector of input values, and $t$ is the target output value. $\eta$ is the learning rate (e.g., .05).

- Initialize each $w_i$ to some small random value
- Until the **termination condition** is met, Do
  - Initialize each $\Delta w_i$ to zero.
  - For each $\langle \vec{x} , t \rangle$ in training examples, Do
    - Input the instance $\vec{x}$ to the unit and compute the output $o$
    - For each linear unit weight $w_i$, Do
```math
\Delta w_i \leftarrow \Delta w_i + \eta (t - o)x_i
```
  - For each linear unit weight $w_i$, Do
```math
w_i \leftarrow w_i + \Delta w_i
```
> For termination condition, it typically is either the error is so small that it is irrelevant or the max step is met, or both.

Linear unit training rule use gradient descent

- Guaranteed to converge to hypothesis with minimum squared error
- Given sufficiently small $\eta$
- Even when training data contains noise
- Even when training data not separable by $H$

### Incremental (Stochastic) Gradient Descent

**Batch mode** Gradient Descent:

Do until satisfied:
1. Compute the gradient $\nabla E_D [\vec{w}]$
2. $\vec{w} \leftarrow \vec{w} - \eta \nabla E_D [\vec{w}]$

**Incremental mode** Gradient Descent:

Do until satisfied:
- For each training example $d$ in $D$
  1. Compute the gradient $\nabla E_d [\vec{w}]$
  2. $\vec{w} \leftarrow \vec{w} - \eta \nabla E_d [\vec{w}]$

> For Deep Learning, there is also another practice called "**mini-batch**", where we segment the training data into small batches. This makes it faster in training, and will be less likely to succumb to messy gradient like Incremental mode.

### Multilayer Networks of Sigmoid Units

<img width="475" alt="Screenshot 2568-03-24 at 16 49 39" src="https://github.com/user-attachments/assets/4f82c879-3eec-4ce7-aa0c-fb5c41e12b0f" />

<img width="349" alt="Screenshot 2568-03-24 at 16 50 05" src="https://github.com/user-attachments/assets/ff3c4190-66c4-4984-9dc3-f54017539879" />

> The task is to recognise vowels. The inputs are F1 and F2 frequencies.

#### Sigmoid Units

<img width="667" alt="Screenshot 2568-03-24 at 16 51 37" src="https://github.com/user-attachments/assets/634fdfd6-df08-4ad2-9bd3-56d6add61bd7" />

$\sigma (x)$ is the sigmoid function

```math
\frac{1}{1 + e^{-x}}
```

Nice property: $\frac{d \sigma (x)}{dx} = \sigma (x) (1 - \sigma (x))$

- By "nice" means that it looks nice to know that a diff'd function is itself multiplied by 1 - itself.

We can derive gradient descent rules to train:

- One sigmoid unit
- Multilayer network of Sigmoid Units &rarr; **Backpropagation**

But how do we know how many nodes to use?

- **Occam's Razor**: try to keep it as minimal as possible.

As for gradient descent:

```math
\text{Error Gradient : } \frac{\partial E}{\partial w_i} = - \sum_{d \in D} (t_d - o_d) o_d (1 - o_d) x_{i,d}
```

### Backpropagation Algorithm

Initialise all weights to small random numbers.
Until satisfied, Do
- For each training example, Do
1. Input the training example to the network and compute the network outputs
2. For each output unit $k$
```math
\delta_k \leftarrow \delta_k (1 - o_k) (t_k - o_k)
```
  3. For each hidden unit $h$
```math
\delta_h \leftarrow o_h (1 - o_h) \sum_{k \in outputs} w_{h,k} \delta_k
```
  4. Update each network weight $w_{i,j}$
```math
w_{i,j} \leftarrow w_{i,j} + \Delta w_{i,j}
```
  where
```math
\Delta w_{i,j} = \eta \delta_j x_{i,j}
```
    
<img width="615" alt="Screenshot 2568-03-24 at 17 05 36" src="https://github.com/user-attachments/assets/4b6762a2-cae3-474e-a2a7-73417106c5e4" />

> This is how a neural network normally works.

some interesting issues:

- Gradient Descent over entrie network weight vector
- Easily generalised to any architecture
- Non-convex
  - Still works well in practice. (It can be run many times with different initialisation.)
- Often include weight momentum $\alpha$
```math
\Delta w_{i,j} (n) = \eta \delta_i x_{i,j} + \alpha \Delta w_{i,j} (n-1)
```
- Minimises error over training examples
  - Same old problems: will it generalise?
- Training can take thousands of iterations &rarr; slow!
- Using network after training (inference) is very fast!

### Learning Hidden Layer Representations

<img width="567" alt="Screenshot 2568-03-24 at 17 23 05" src="https://github.com/user-attachments/assets/0314a34d-504d-43f2-b3e2-0c0df15a5506" />

Can this be learned? (Yeah)

<img width="547" alt="Screenshot 2568-03-24 at 17 23 29" src="https://github.com/user-attachments/assets/e615a279-a473-4d51-98bd-1525c4ae3397" />

If we look closely:

<img width="486" alt="Screenshot 2568-03-24 at 17 24 18" src="https://github.com/user-attachments/assets/c1487d1f-0f28-44fe-8855-d38a7569938f" />

- we can see that the hidden values, if rounded are bascially 3 bits representation!

#### Training

<img width="483" alt="Screenshot 2568-03-24 at 17 25 27" src="https://github.com/user-attachments/assets/c2d7db41-d131-4c35-b19b-7b5b58c0ddf2" />

<img width="479" alt="Screenshot 2568-03-24 at 17 25 57" src="https://github.com/user-attachments/assets/88366d9a-4bed-4644-a001-3c9c9bc49c28" />

<img width="450" alt="Screenshot 2568-03-24 at 17 26 18" src="https://github.com/user-attachments/assets/3088cec2-f4da-4f85-b407-4103b181b804" />

#### Convergence of Backpropagation

Gradient descent to some local minimum
- Perhaps not global
- Try add **momentum**
- SGD!
- Train multiple networks with different initial weights

Nature of convergence
- Initilaise weights near zero (don't initialise with big values)
- Therefore, initial networks near-linear
- Increasingly non-linear functions possible as training progresses

## Expressive Capabilities of ANNs

Boolean functions:
- Every boolean function can be represented by network with a single hidden layer (but maybe many nodes)

Continuous functions:
- Every bounded continuous function can be approximated with small error with just one hidden layer (again with immense number of nodes)
- Any function can be approximated to almost perfect accuracy by a network with two hidden layers

## Overfitting in ANNs

Like any model with impressive ability to fit to any data, ANNs tend to overfit to the training data.

<img width="411" alt="Screenshot 2568-03-24 at 21 34 37" src="https://github.com/user-attachments/assets/84dcedeb-68ba-4fae-908a-932e69969010" />

In a long run, errors will always decrease, but once it is converged any additional training will cause the error to go up. Therefore it is highly recommended to always monitor the error graph to see if the error has started rising.

<img width="420" alt="Screenshot 2568-03-24 at 21 34 48" src="https://github.com/user-attachments/assets/f6aba5d3-6330-4c7e-9db0-ea0af28b1fb7" />

Additionally, make sure to not look at the graph at a window too small, so that we don't mistake a local minimum as a global one

## Alternative Error Functions

We can penalise large weights (like regularisation in Linear Regression and Logistic Regression)

```math
E (\vec{w}) \equiv \frac{1}{2} \sum_{d \in D} \sum_{k \in outputs} (t_{kd} - o_{kd})^2 + \gamma \sum_{i,j} w_{ji}^{2}
```
> Here we add regularisation term: $+ \gamma \sum_{i,j} w_{ji}^{2}$ to penalise the large weights

Or we can train on slopes as well

```math
E (\vec{w}) \equiv \frac{1}{2} \sum_{d \in D} \sum_{k \in outputs} \left[ (t_{kd} - o_{kd})^2 + \mu \sum_{j \in inputs} \left( \frac{\partial t_{kd}}{\partial x_{d}^{j}} - \frac{\partial o_{kd}}{\partial x_{d}^{j}} \right)^{2} \right]
```
> Here we do not focus on calculating errors by comparing output values with target values, but we compare their slope (in error graph as well).

We can tie weights together as well! (e.g., like in CNN, the convolutional filter is this exact tied weights)

## Recurrent Networks

Unlike normal feedforward networks, the Recurrent Networks uses the output of a time step to be the input of the next.

<img width="349" alt="Screenshot 2568-03-24 at 21 46 38" src="https://github.com/user-attachments/assets/b04b1b03-b353-4eb7-b3cc-1359f2d2d7df" />

Still don't get it? Here's the visualisation as mimic feedforward

<img width="280" alt="Screenshot 2568-03-24 at 21 48 57" src="https://github.com/user-attachments/assets/095dc91c-3886-439a-994b-f44079b412c9" />
