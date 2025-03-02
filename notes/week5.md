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
x = \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ . \\ . \\ . \\ x_n \end{bmatrix} \in R^{n+1} \quad \theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \theta_2 \\ . \\ . \\ . \\ \theta_n \end{bmatrix} \in R^{n+1}
```

Looking at the terms this way, we can put it in a shorter, more elegant term as:

$$ h_\theta(x) = \theta^{\top} x $$

While, $\theta^{\top} x$ is basically $\theta x$, but in finding dot product of two vectors, we need to make sure to transpose the leading vector in order to find them.

```math
[\theta_0, \theta_1, \theta_2,\cdots, \theta_n] \begin{bmatrix} x_0 \\ x_1 \\ x_2 \\ . \\ . \\ . \\ x_n \end{bmatrix} \in R^{n+1}
```

> Linear Regression with Multiple Variables is also known as Multivariate Linear Regression.

## Gradient Descent for multiple variables

Hypothesis: $h_\theta(x) = \theta^{\top} x = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n$

Parameters: $\theta = [\theta_0, \theta_1, \cdots, \theta_n]^{\top}$

Cost Function:

$$ J(\theta) = J(\theta_0, \theta_1, \cdots, \theta_n) = \frac{1}{2m} \sum_{i+1}^{m} (h_\theta(x^{(i)} - y^{(i)}))^2 $$

Gradient Descent:

$\text{With}$

$$ J(\theta) = J(\theta_0, \theta_1, \cdots, \theta_n) \text{ ,} $$

$\text{repeat } \\{$

$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) \quad (\text{simultaneously update for every } j = 0, \cdots, n) $$

$\\}$

Previously ($n = 1$):

# Logistic Regression
