# Linear Regression with Multiple Variables

## Multiple features

### Multiple features (variables).

Before, we only deal with one single feature (Size) to predict the target value (Price).

| $x$ : Size (feet<sup>2</sup>) | $y$ : Price ($1000) |
| :--: | :---: |
| 2104 | 460 |
| 1416 | 232 |
| 1534 | 315 |
| 852 | 178 |
| ... | ... |

Target function:

$$ h_\theta(x) = \theta_0 + \theta_1 x $$

Now, we include three more features (Number of bedrooms, Number of floors, and Age of home) with hope that more information will improve the model accuracy.

| $x_1$ : Size (feet<sup>2</sup>) | $x_2$ : Number of bedrooms | $x_3$ : Number of floors | $x_4$ : Age of home (years) | $y$ : Price ($1000) |
| :--: | :---: | :---: | :---: | :---: |
| 2104 | 5 | 1 | 45 | 460 |
| 1416 | 3 | 2 | 40 | 232 |
| 1534 | 3 | 2 | 30 | 315 |
| 852 | 2 | 1 | 36 | 178 |
| ... | ... | ... | ... | ... |

notation:

- $n$ = number of features
  - In this case, $n = 4$, while $m$ which is the number of datapoints in the dataset is still 47.
- $x^{(i)}$ = input (features) of $i^{th}$ training example
  - An instance, say $x^{(2)}$, which used to be a scalar value $x^{(2)} = 1416$ will now be a vector:

```math
x^{(2)} = \begin{bmatrix} 1416 \\ 3 \\ 2 \\ 40 \end{bmatrix}
```

- $x_{j}^{(i)}$ = value of feature $j$ in $i^{th}$ training example
  - For example, $x_{3}^{(2)} = 2$

Hypothesis:

- Previously:

$$ h_\theta(x) = \theta_0 + \theta_1 x $$

- Now: 

$$ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \theta_4 x_4 $$

or more broadly,:

$$ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n $$

In actual implementation, we will also add $x_0 = 1$ (or specifically $x_{0}^{(i)} = 1$), into the features as well, to avoid programming complications. Therefore, the hypothesis will be: 

$$h_\theta(x) = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \theta_4 x_4 \quad (x_0 \text{ is always } 1)$$

We are now working with multi-dimensional value, will operate accordingly.

$$ h_\theta(x) = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n $$

both $x$ and $\theta$ are now vectors. (although $\theta$ has always secretly been a vector.)
 
```math
x = \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ . \\ . \\ . \\ x_n \end{bmatrix} \in ℝ^{n+1} \quad \theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \theta_2 \\ . \\ . \\ . \\ \theta_n \end{bmatrix} \in ℝ^{n+1}
```

Looking at the terms this way, we can put it in a shorter, more elegant term as:

$$ h_\theta(x) = \theta^{\top} x $$

While, $\theta^{\top} x$ is basically $\theta x$, but in finding dot product of two vectors, we need to make sure to transpose the leading vector in order to find them.

```math
[\theta_0, \theta_1, \theta_2,\cdots, \theta_n] \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ . \\ . \\ . \\ x_n \end{bmatrix} \in ℝ^{n+1}
```

> Linear Regression with Multiple Variables is also known as Multivariate Linear Regression.

## Gradient Descent for multiple variables

Hypothesis: $h_\theta(x) = \theta^{\top} x = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$

Parameters: $\theta = [\theta_0, \theta_1, \cdots, \theta_n]^{\top}$

Cost Function:

$$ J(\theta) = J(\theta_0, \theta_1, \cdots, \theta_n) = \frac{1}{2m} \sum_{i+1}^{m} (h_\theta(x^{(i)} - y^{(i)}))^2 $$

Gradient Descent:

$\text{Repeat } \\{$

$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} \overbrace{J(\theta)}^{J(\theta_0, \theta_1, \cdots, \theta_n)} \quad (\text{simultaneously update for every } j = 0, \cdots, n) $$

$\\}$

Previously ($n = 1$):

$\text{Repeat } \\{$

$$ \theta_0 := \theta_0 - \alpha \underbrace{\frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})}_{\frac{\partial}{\partial \theta_0} J(\theta)} $$

$$ \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x^{(i)} $$

$$ (\text{simultaneously update } \theta_0, \theta_1) $$

$\\}$

New algorithm ($n \geq 1$):

$\text{Repeat } \\{$

$$ \theta_j := \theta_j - \alpha \overbrace{\frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_j^{(i)}}^{\frac{\partial}{\partial \theta_0} J(\theta)} \quad (\text{simultaneously update for every } j = 0, \cdots, n) $$

$\\}$

So, for each $\theta_n$:

$$ \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \overbrace{x_{0}^{(i)}}^{x_{0}^{(i)} = 1} $$

$$ \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_{1}^{(i)} $$

$$ \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_{2}^{(i)} $$

$$ \cdots $$

## Gradient Descent in practice

### Future Scaling

Idea: Make sure features are on a similar scale.

For example:
- $x_1$ = size (0-2000 feet<sup>2</sup>)
- $x_2$ = number of bedrooms (1-5)

<img width="276" alt="Screenshot 2568-03-03 at 02 32 31" src="https://github.com/user-attachments/assets/ef144dc9-da43-48a7-9bc7-4d9504a92b48" />

$$ x_1 = \frac{\text{size (feet)}^2}{2000} $$

$$ x_2 = \frac{\text{number of bedrooms}}{5} $$

Now $x_1$ is $0 \leq x_1 \leq 1$ and $x_2$ is $0 \leq x_2 \leq 1$.

<img width="245" alt="Screenshot 2568-03-03 at 02 38 06" src="https://github.com/user-attachments/assets/8e5d8fb1-8edf-4e30-bf45-ea16b314f90a" />

**Future Scaling**

Now, get every feature into approximately a $-1 \leq x_i \leq 1$ range.

<img width="541" alt="Screenshot 2568-03-03 at 02 39 47" src="https://github.com/user-attachments/assets/031d88c7-d635-4aff-8a7b-a977fa7435ca" />

**Mean normalisation**

Replace $x_i$ with $x_i - \mu_i$ to make features have approximately zero mean (Do not apply to $x_0 = 1$).

For example:

$$ x_1 = \frac{size - 1000}{2000} $$

$$ x_2 = \frac{\\# bedrooms - 2}{5} $$

$$ -0.5 \leq x_1 \leq 0.5, -0.5 \leq x_2 \leq 0.5 $$

$$ x_1 \leftarrow \frac{x_1 - \overbrace{\mu_1}^{\text{avg value of } x_1 \text{ in training set}}}{\underbrace{s_i}_{\text{range (max-min) or standard deviation}}} $$

$$x_2 \leftarrow \frac{x_2 - \mu_2}{s_i}$$

### Learning rate

**Gradient descent**

$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) $$

- "Debugging": How to make sure gradient descent is working correctly.
- How to choose learning rate $\alpha$.

**Making sure gradient descent is working correctly.**

<img width="585" alt="Screenshot 2568-03-03 at 02 53 22" src="https://github.com/user-attachments/assets/2b722b07-fab9-4694-9159-8f4741de97df" />

<img width="557" alt="Screenshot 2568-03-03 at 02 53 49" src="https://github.com/user-attachments/assets/ccdee0ee-e13a-4fbf-83b8-7a4cfded6a15" />

- For sufficiently small $\alpha$, $J(\theta)$ should decrease on every iteration.
- but if $\alpha$ is too small, gradient descent can be slow to converge.

**Summary**:

- If $\alpha$ is too small: slow convergence.
- If $\alpha$ is too large: $J(\theta)$ may not converge. (slow converge also possible)

To choose $\alpha$, try:

$$ \cdots, 0.001 \underset{3 \times}{,} 0.003 \underset{\approx 3 \times}{,} 0.01 \underset{3 \times}{,} 0.03 \underset{\approx 3 \times}{,} 0.1 \underset{3 \times}{,} 0.3 \underset{\approx 3 \times}{,} 1, \cdots $$

## Features and polynomial regression

**Housing price prediction**

$$ h_\theta(x) = \theta_0 + \theta_1 \times \underbrace{frontage}\_{x_1} + \theta_2 \times \underbrace{depth}\_{x_2} $$

> **Area**
> 
> $$ x = frontage \times depth $$
> 
> $$ h_\theta(x) = \theta_0 + \theta_1 \underbrace{x}_{\text{land area}} $$

**Polynomial regression**

<img width="618" alt="Screenshot 2568-03-03 at 03 13 49" src="https://github.com/user-attachments/assets/7d40b125-6a51-457d-9b49-b34ae7d34e11" />

$$ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 $$ 

$$ = \theta_0 + \theta_1 (size) + \theta_2 (size)^2 + \theta_3 (size)^3 $$

$$ x_1 = (size) $$

$$ x_2 = (size)^2 $$

$$ x_3 = (size)^3 $$

**Choice of features**

<img width="479" alt="Screenshot 2568-03-03 at 03 11 01" src="https://github.com/user-attachments/assets/d3322a6d-404f-4c27-b442-3d770746a7d7" />

$$ h_\theta(x) = \theta_0 + \theta_1 (size) + \theta_2 (size)^2 $$

$$ h_\theta(x) = \theta_0 + \theta_1 (size) + \theta_2 \sqrt{(size)} $$

## Normal Equation

Gradient Descent

<img width="274" alt="Screenshot 2568-03-03 at 03 22 58" src="https://github.com/user-attachments/assets/79d501b3-fae8-4ded-8d74-e83080dbb0c0" />

Normal equation: Method to solve for $\theta$ analytically.

Intuition: If 1D ($\theta \in ℝ$)

$$ J(\theta) = a \theta^2 + b \theta + c $$

<img width="205" alt="Screenshot 2568-03-03 at 03 28 57" src="https://github.com/user-attachments/assets/fc81b04d-14f2-42c9-b763-93f7f85f9ff7" />

$$ \theta \in ℝ^{n+1} \quad J(\theta_0, \theta_1, \cdots, \theta_m) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 $$

$$ \frac{\partial}{\partial \theta_j} J(\theta) = \cdots \overset{set}{=} 0 \quad (\text{for every } j) $$

$$ \text{Solve for } \theta_0, \theta_1, \cdots, \theta_n $$

Examples: $m = 4$.

| $x_0$ | $x_1$ : Size (feet<sup>2</sup>) | $x_2$ : Number of bedrooms | $x_3$ : Number of floors | $x_4$ : Age of home (years) | $y$ : Price ($1000) |
| :--: | :--: | :---: | :---: | :---: | :---: |
| 1 | 2104 | 5 | 1 | 45 | 460 |
| 1 | 1416 | 3 | 2 | 40 | 232 |
| 1 | 1534 | 3 | 2 | 30 | 315 |
| 1 | 852 | 2 | 1 | 36 | 178 |

```math
X = {\begin{bmatrix} 1 & 2104 & 5 & 1 & 45 \\ 1 & 1416 & 3 & 2 & 40 \\ 1 & 1534 & 3 & 2 & 30 \\ 1 & 852 & 2 & 1 & 36 \end{bmatrix}}^{m \times (n + 1)} \qquad y = \begin{bmatrix} 460 \\ 232 \\ 315 \\ 178 \end{bmatrix}^{m \text{ dimensional vector}}
```

$$ \theta = (X^{\top} X)^{-1} X^{\top} y $$

$m$ examples $(x^{(1)}, y^{(1)}), \cdots, (x^{(m)}, y^{(m)})$; $n$ features.

```math
x^{(i)} = \begin{bmatrix} x_{0}^{(i)} \\ x_{1}^{(i)} \\ x_{2}^{(i)} \\ . \\ . \\ . \\ x_{n}^{(i)} \end{bmatrix} \in ℝ^{n + 1} \qquad \underset{\text{(design matrix)}}{X} = \begin{bmatrix} \textemdash \textemdash & (x^{(1)})^{\top} & \textemdash \textemdash \\ \textemdash \textemdash & (x^{(2)})^{\top} & \textemdash \textemdash \\  & . &  \\  & . &   \\  & . & \\ \textemdash \textemdash & (x^{(m)})^{\top} & \textemdash \textemdash \end{bmatrix}
```

E.g. If 

```math
x^{(i)} = \begin{bmatrix} 1 \\ x_{1}^{(i)} \end{bmatrix} \underset{\theta = (X^{\top} X)^{-1} X^{\top} y}{\longrightarrow} X = \begin{bmatrix} 1 & x_{1}^{(i)} \\ 1 & x_{2}^{(i)} \\ . & . \\  . & . \\  . & . \\ 1 & x_{m}^{(i)} \end{bmatrix}^{m \times 2} \qquad y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ . \\ . \\ . \\ y^{(m)} \end{bmatrix}
```

$$ \theta = (X^{\top} X)^{-1} X^{\top} y $$

$(X^{\top} X)^{-1}$ is inverse of matrix $X^{\top} X$.

$$ \{Set } A = X^{\top} X $$

$$ (X^{\top} X)^{-1} = A^{-1} $$

Octave: 

$$ pinv (X^\prime X) X^\prime y $$

$$ pinv (X^{\top} X) X^{\top} y $$

$$ \theta = (X^{\top} X)^{-1} X^{\top} y $$

$m$ training examples, $n$ features.

| Gradient Descent | Normal Equation |
| --- | --- |
| - Need to choose $\alpha$. <br> - Needs many iterations. <br> - Works well even when $n$ is large. | - No need to choose $\alpha$. <br> - Don't need to iterate. <br> - Need to compute $(X^{\top} X)^{-1}$ <br> - Slow if $n$ is very large. |

## Normal equation and non-invertability (optional)

Normal equation

$$ \theta = (X^{\top} X)^{-1} X^{\top} y $$

- What if $X^{\top} X$ is non-invertible? (singular/degenerate)
- Octave: $pinv (X^\prime X) X^\prime y$ ($\theta$)

What if $X^{\top} X$ is non-invertible?

- Redundant features (linearly dependent).
  - E.g., $x_1$ = size in feet<sup>2</sup> and $x_2$ size in m<sup>2</sup>
- Too many features (e.g., $m \leq n$).
  - Delete some features, or use regularisation.

# Logistic Regression

## Classification

Given an instance (a datapoint with information as values of different features/attrubutes.), let a machine (the ML model) try to give a "label" to that instance.

Example use case:
- Email: Spam/Not Spam
- Online: Transactions: Fraudulent (Yes/No)?
- Tumour: Malignant/Benign

These are examples of a subcategoty of classification task known as "binary classification". (Two labels: 0/1, yes/no)

$$ y \in \\{\underset{0:\text{"Negative Class (e.g., benign tumour)"}}{0}, \underset{1:\text{"Positive Class (e.g., malignant tumour)"}}{1}\\} $$

If there are more than two labels, this subcategory is called "multiclass classification".

$$ y \in \\{0, 1, 2, 3\\} $$

Like Linear Regression, Logistic Regression implements linear function as its target function. However, this doesn't mean that you can use the function to "classify" right away.

<img width="626" alt="Screenshot 2568-03-03 at 16 14 08" src="https://github.com/user-attachments/assets/dc4663f1-657c-40d2-88d6-21a69fbae09f" />

If we use pure linear function like the one we use in Linear Regression, it will output put a continuous value (let's call it $\hat{y}$) that is: $0 \leq hat{y} \leq 1$. This is because the classes are depicted to the model as numbers (0,1), making the lowest possible label to be 0 and highest 1. 

This is not ideal. Given that the outputs are continuos values $\hat{y}$ that are $0 \leq hat{y} \leq 1$, if we plot the linear function onto the graph, we will get a straight line, and give the model another instance to train on and plot the graph again, the straight line will change drastically.

**How do we fix this?**:

Threshold classifier output $h_\theta(x)$ at o.5:
- If $h_\theta(x) \geq 0.5$, predict "y=1"
- If $h_\theta(x) < 0.5$, predict "y=0"

Classification: $y = 0 \text{ or } 1$  
$h_\theta(x)$ can be $>1$ or $<0$

$$ \underset{\text{the task is classfication (predict discreet value), not regression (predict continuous value)}}{\text{Logistic Regression}}: 0 \leq h_\theta(x) \leq 1 $$

## Hypothesis representation

**Logistic Regression Model**

Find $0 \leq h_\theta(x) \leq 1$

linear regression: $h_\theta(x) = \theta^{\top} x$

How do we keep $h_\theta(x)$ between 0 and 1 if the encoded labels range more than that? We use "Logistic Function" (Here we use "Sigmoid Function" $g(z)$ ).

$$ g(z) = \frac{1}{1 + e^{-z}} $$

$$ h_\theta(x) = g(\theta^{\top} x) = \frac{1}{1 + e^{-\theta^{\top} x}} $$

<img width="312" alt="Screenshot 2568-03-03 at 19 26 36" src="https://github.com/user-attachments/assets/82a68289-1182-47db-b503-25b338764413" />

**Interpretation of Hypothesis Output**

$h_\theta(x)$ = estimated probability that $y = 1$ on input x

Example: If

```math
x = \begin{bmatrix} x_0 \\ x_1 \end{bmatrix} = \begin{bmatrix} 1 \\ tumourSize \end{bmatrix}
```

$$ h_theta(x) = 0.7 $$

Tell the patient that the chance of the tumour being malignant is 70%.

"probability that $y = 1$, given $x$, parameterised by $\theta$"

$$ \text{Given the nature of probability}: P(y = 0) + P(y = 1) = 1 $$

$$ P(y = 0 \mid x ; \theta) = 1 - P(y = 1 \mid x ; \theta) $$

Therefore,

$$ h_\theta(x) = P(y = 1 \mid x ; \theta) $$

## Decision Boundary

**Logistic Regression**

$$ h_\theta(x) = g(\theta^{\top} x) $$

$$ g(z) = \frac{1}{1 + e^{-z}} $$

Suppose:

predict " $y = 1$ if $h_\theta(x) \geq 0.5$ "

predict " $y = 0$ if $h_\theta(x) < 0.5$ "

