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







