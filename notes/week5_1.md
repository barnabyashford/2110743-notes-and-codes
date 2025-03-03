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

E.g.,

$$ h_\theta(x) = 80 + 0.1 x_1 + 0.01 x_2 + 3 x_3 + 2 x_4  $$

So, we can also say that $\theta$'s are weights of each feature.

or more broadly,:

$$ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n $$

In actual implementation, we will also add $x_0 = 1$ (or specifically $x_{0}^{(i)} = 1$), into the features as well, to avoid programming complications. Therefore, the hypothesis will be: 

$$h_\theta(x) = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 + \theta_4 x_4 \quad (x_0 \text{ is always } 1)$$

We are now working with multi-dimensional value, will operate accordingly.

$$ h_\theta(x) = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n = \sum_{i=0}^{n} \theta_i x_i $$

both $x$ and $\theta$ are now vectors. (although $\theta$ has always secretly been a vector.)
 
```math
x = \begin{bmatrix} x_0 = 1 \\ x_1 \\ x_2 \\ . \\ . \\ . \\ x_n \end{bmatrix} \in ℝ^{n+1} \quad \theta = \begin{bmatrix} \theta_0 \\ \theta_1 \\ \theta_2 \\ . \\ . \\ . \\ \theta_n \end{bmatrix} \in ℝ^{n+1}
```

Looking at the terms this way, we can put it in a shorter, more elegant term as:

$$ h_\theta(x) = \theta^{\top} x $$

or if we think of $\theta$ and $x$ as vectors we can just write:

$$ h_\theta(x) = \theta x $$

$\theta^{\top} x$ is basically $\theta x$ from the view of matrix operation. In finding dot product of two matrices, we need to make sure to transpose the leading matrix in order to find them.

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

### Feature Scaling

Idea: Make sure features are on a similar scale.

For example:
- $x_1$ = size (0-2000 feet<sup>2</sup>)
- $x_2$ = number of bedrooms (1-5)

<img width="276" alt="Screenshot 2568-03-03 at 02 32 31" src="https://github.com/user-attachments/assets/ef144dc9-da43-48a7-9bc7-4d9504a92b48" />

>  This is what happens when we plot contour graph without scaling our features. When use GD with the graph like this, it will take a lot of time.

**Divide by max**

$$ x_1 = \frac{\text{size (feet)}^2}{2000} $$

$$ x_2 = \frac{\text{number of bedrooms}}{5} $$

Now $x_1$ is $0 \leq x_1 \leq 1$ and $x_2$ is $0 \leq x_2 \leq 1$.

<img width="245" alt="Screenshot 2568-03-03 at 02 38 06" src="https://github.com/user-attachments/assets/8e5d8fb1-8edf-4e30-bf45-ea16b314f90a" />

>  This is what happens when we plot contour graph after scaling our features. When use GD with the graph like this, it will take significantly less time.

Earilier we scaled the features in range $0 \leq x_i \leq 1$ . We can also get every feature into approximately a $-1 \leq x_i \leq 1$ range.

<img width="541" alt="Screenshot 2568-03-03 at 02 39 47" src="https://github.com/user-attachments/assets/031d88c7-d635-4aff-8a7b-a977fa7435ca" />

> This image shows that it's okay if the feature is already ranging from small negative number to small positive number, and if the range if as big as hundreds or thousands, we really have to scale it.

**Mean normalisation**

Replace $x_i$ with $x_i - \mu_i$ to make features have approximately zero mean (Do not apply to $x_0 = 1$).

For example:

$$ x_1 = \frac{size - 1000}{2000} $$

$$ x_2 = \frac{\\# bedrooms - 2}{5} $$

$$ -0.5 \leq x_1 \leq 0.5, -0.5 \leq x_2 \leq 0.5 $$

**$z$-normalisation**

$$ x_1 \leftarrow \frac{x_1 - \overbrace{\mu_1}^{\text{avg value of } x_1 \text{ in training set}}}{\underbrace{s_1}_{\text{range (max-min) or standard deviation}}} $$

$$x_2 \leftarrow \frac{x_2 - \mu_2}{s_2}$$

### Learning rate

**Gradient descent**

$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) $$

- "Debugging": How to make sure gradient descent is working correctly.
- How to choose learning rate $\alpha$.

**Making sure gradient descent is working correctly.**

<img width="585" alt="Screenshot 2568-03-03 at 02 53 22" src="https://github.com/user-attachments/assets/2b722b07-fab9-4694-9159-8f4741de97df" />

> This is how the error graphs in each iteration should be.

<img width="557" alt="Screenshot 2568-03-03 at 02 53 49" src="https://github.com/user-attachments/assets/ccdee0ee-e13a-4fbf-83b8-7a4cfded6a15" />

> If the error graph is like any of this, something might be wrong

- For sufficiently small $\alpha$, $J(\theta)$ should decrease on every iteration.
- but if $\alpha$ is too small, gradient descent can be slow to converge.

**Summary**:

- If $\alpha$ is too small: slow convergence.
- If $\alpha$ is too large: $J(\theta)$ may not converge. (slow converge also possible)

To choose $\alpha$, try:

$$ \cdots, 0.001 \underset{3 \times}{,} 0.003 \underset{\approx 3 \times}{,} 0.01 \underset{3 \times}{,} 0.03 \underset{\approx 3 \times}{,} 0.1 \underset{3 \times}{,} 0.3 \underset{\approx 3 \times}{,} 1, \cdots $$

## Features and polynomial regression

**Housing price prediction**

We might use size as feature, but all we have are $frontage$ and $depth$. So, it might go this way if we don't do something with that:

$$ h_\theta(x) = \theta_0 + \theta_1 \times \underbrace{frontage}\_{x_1} + \theta_2 \times \underbrace{depth}\_{x_2} $$

Separately, $frontage$ and $depth$ might not mean much, and it might be unnecessary to compute TWO weights for two features. However, we know that $frontage$ and $depth$ can be used to compute land area, so here's what we do:

$$ x = frontage \times depth $$

$$ h_\theta(x) = \theta_0 + \theta_1 \underbrace{x}_{\text{land area}} $$

**Polynomial regression**

<img width="618" alt="Screenshot 2568-03-03 at 03 13 49" src="https://github.com/user-attachments/assets/7d40b125-6a51-457d-9b49-b34ae7d34e11" />

Sometimes, our data distribution may not be a straight, but a curve line, instead of vanilla linear regression, we can try: "Polynomial Regression"

$$ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_3 $$ 

$$ = \theta_0 + \theta_1 (size) + \theta_2 (size)^2 + \theta_3 (size)^3 $$

$$ x_1 = (size) $$

$$ x_2 = (size)^2 $$

$$ x_3 = (size)^3 $$

**Choice of features**

<img width="479" alt="Screenshot 2568-03-03 at 03 11 01" src="https://github.com/user-attachments/assets/d3322a6d-404f-4c27-b442-3d770746a7d7" />

We can also use square roots as well.

$$ h_\theta(x) = \theta_0 + \theta_1 (size) + \theta_2 (size)^2 $$

$$ h_\theta(x) = \theta_0 + \theta_1 (size) + \theta_2 \sqrt{(size)} $$

## Normal Equation

Gradient Descent

<img width="274" alt="Screenshot 2568-03-03 at 03 22 58" src="https://github.com/user-attachments/assets/79d501b3-fae8-4ded-8d74-e83080dbb0c0" />

Theoretically, if the function maps to a parabola function, which has only one lowest point (surely a global mimimum), we can find it without having to use GD as well.

Normal equation: Method to solve for $\theta$ analytically.

Intuition: If 1D ($\theta \in ℝ$)

$$ J(\theta) = a \theta^2 + b \theta + c $$

With GD, to find the next step, we differentiate the equation to find the slope. Here we can fix it to 0, as we know that there's only one point that the slope is zero: the global minimum, then we solve the equation for the values inside.

$$ \frac{d}{d \theta} J(\theta) = \cdots \overset{\text{set}}{=} 0 $$

$$ \text{Solve for } \theta $$

<img width="205" alt="Screenshot 2568-03-03 at 03 28 57" src="https://github.com/user-attachments/assets/fc81b04d-14f2-42c9-b763-93f7f85f9ff7" />

$$ \theta \in ℝ^{n+1} \quad J(\theta_0, \theta_1, \cdots, \theta_m) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 $$

$$ \frac{\partial}{\partial \theta_j} J(\theta) = \cdots \overset{set}{=} 0 \quad (\text{for every } j) $$

$$ \text{Solve for } \theta_0, \theta_1, \cdots, \theta_n $$

Examples: $m = 4$, $n = 4$.

| $x_0$ | $x_1$ : Size (feet<sup>2</sup>) | $x_2$ : Number of bedrooms | $x_3$ : Number of floors | $x_4$ : Age of home (years) | $y$ : Price ($1000) |
| :--: | :--: | :---: | :---: | :---: | :---: |
| 1 | 2104 | 5 | 1 | 45 | 460 |
| 1 | 1416 | 3 | 2 | 40 | 232 |
| 1 | 1534 | 3 | 2 | 30 | 315 |
| 1 | 852 | 2 | 1 | 36 | 178 |

```math
X = {\begin{bmatrix} 1 & 2104 & 5 & 1 & 45 \\ 1 & 1416 & 3 & 2 & 40 \\ 1 & 1534 & 3 & 2 & 30 \\ 1 & 852 & 2 & 1 & 36 \end{bmatrix}}^{m \times (n + 1)} \qquad y = \begin{bmatrix} 460 \\ 232 \\ 315 \\ 178 \end{bmatrix}^{m \text{ dimensional vector}}
```

$$ \text{Solved: } \theta = (X^{\top} X)^{-1} X^{\top} y $$

$m$ examples $(x^{(1)}, y^{(1)}), \cdots, (x^{(m)}, y^{(m)})$; $n$ features.

```math
x^{(i)} = \begin{bmatrix} x_{0}^{(i)} \\ x_{1}^{(i)} \\ x_{2}^{(i)} \\ . \\ . \\ . \\ x_{n}^{(i)} \end{bmatrix} \in ℝ^{n + 1} \qquad \underset{\text{(design matrix)}}{X} = \begin{bmatrix} \textemdash \textemdash & (x^{(1)})^{\top} & \textemdash \textemdash \\ \textemdash \textemdash & (x^{(2)})^{\top} & \textemdash \textemdash \\  & . &  \\  & . &   \\  & . & \\ \textemdash \textemdash & (x^{(m)})^{\top} & \textemdash \textemdash \end{bmatrix}
```

E.g. If 

```math
x^{(i)} = \begin{bmatrix} 1 \\ x_{1}^{(i)} \end{bmatrix}
```

then $X$ and $y$ would be:

```math
X = \begin{bmatrix} 1 & x_{1}^{(1)} \\ 1 & x_{1}^{(2)} \\ . & . \\  . & . \\  . & . \\ 1 & x_{1}^{(m)} \end{bmatrix}^{m \times n +1} \quad y = \begin{bmatrix} y^{(1)} \\ y^{(2)} \\ . \\ . \\ . \\ y^{(m)} \end{bmatrix}
```

after getting $X$ and $y$, we can send them straight to the equation $\theta = (X^{\top} X)^{-1} X^{\top} y$.

Some example in implementation:

$$ \theta = (X^{\top} X)^{-1} X^{\top} y $$

$X^{\top} X$ will result in a square matrix, as the dimensions are: $X^{\top}:(n+1) \times m$ and $X: m \times (n+1)$ and therefore, it can be inversed. $(X^{\top} X)^{-1}$ is inverse of matrix $X^{\top} X$.

$$ \{Set } A = X^{\top} X $$

$$ (X^{\top} X)^{-1} = A^{-1} $$

Octave: 

```octave
pinv(X' *X) *X' *y
```
> In Octave, `pinv` is a function to find "pseudoinverse" of a matrix and `X'` is just Octave's version of $X^{\top}$. So, `pinv(X' *X) *X' *y` means $\theta = (X^{\top} X)^{-1} X^{\top} y$

With normal equation, we have no need for tuning the $\alpha$ and hence no need to perform feature scaling.

$m$ training examples, $n$ features.

| Gradient Descent | Normal Equation |
| --- | --- |
| - Need to choose $\alpha$. <br> - Needs many iterations. <br> - Works well even when $n$ is large. | - No need to choose $\alpha$. <br> - Don't need to iterate. <br> - Need to compute $(X^{\top} X)^{-1}$ <br> - Slow if $n$ is very large. |

## Normal equation and non-invertability (optional)

Normal equation

$$ \theta = (X^{\top} X)^{-1} X^{\top} y $$

- What if $X^{\top} X$ is non-invertible? (singular/degenerate)
  - With Octave: `pinv(X' *X) *X' *y`, we can find "pseudoinverse".

**Inverse Matrix vs Pseudoinverse Matrix**

$A$ = Any invertible matrix

$AA^{-1}$ will result in an identity matrix $I$. For example, an identity matrix $I_{\text{example}}$ in dimension of $5 \times 5$

```math
I_{\text{example}} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 1 \end{bmatrix}
```

$D$ = Any non-invertible matrix

$D pinv(D)$ will result in a matrix with its nature similar to that of an identity matrix with small numbers close to 1 instead of just 1.

How can $X^{\top} X$ potentially be non-invertible?

- Redundant features (linearly dependent).
  - E.g., $x_1$ = size in feet<sup>2</sup> and $x_2$ size in m<sup>2</sup>
- Too many features (e.g., $m \leq n$).
  - Delete some features, or use regularisation.
