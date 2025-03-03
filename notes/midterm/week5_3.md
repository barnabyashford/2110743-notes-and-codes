# Regularisation

## The problem of overfitting

<img width="573" alt="Screenshot 2568-03-04 at 01 27 10" src="https://github.com/user-attachments/assets/3ce5fa6c-a26d-4996-8242-2574684e27fa" />

> Example: Linear Regression

**High bias** leads to **underfitting**, limiting the model to reach better performance

**High variance** leads to **overfitting**, it has too much flexibility to fit to training data, and will then fail in downstream implementation.

Overfitting: If we have too many features, the learned hypothesis may fit the training set very well $\left( J(0) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^{2} \approx 0 \right)$, but
fail to generalize to new examples (predict prices on new examples).

We need to give the model the ability to **generalise**.

<img width="601" alt="Screenshot 2568-03-04 at 01 29 53" src="https://github.com/user-attachments/assets/0acdadc7-5239-4dbb-8350-aca551bc6f2e" />

> Example: Logistic Regression

**Addressing overfitting**:

<img width="592" alt="Screenshot 2568-03-04 at 01 30 15" src="https://github.com/user-attachments/assets/16be5ad3-ed55-48d6-a025-936c4d161111" />

Options:

1. Reduce number of features.
  - Manually select which features to keep.
  - Model selection algorithm. (similar to tree pruning in DT, try remove a feature, and test in validation set to see if it performs better.)
2. Regularization.
  - Keep all the features, but reduce magnitude/values of parameters $\theta_j$
  - Works well when we have a lot of features, each of which contributes a bit to predicting $y$.

## Cost Function

**Intuition**

<img width="574" alt="Screenshot 2568-03-04 at 02 38 48" src="https://github.com/user-attachments/assets/a13e3718-8c03-4439-a13a-3e959211a47f" />

Suppose we penalize and make $\theta_3$, $\theta_4$ really small.

$$ \underset{\theta}{min } \frac{1}{2m} \sum_{i = 1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \underbrace{1000\theta_{3}^{2}}\_{\theta_3 \approx 0}+ \underbrace{1000\theta_{4}^{4}}\_{\theta_4 \approx 0} $$

Reducing weights of the unnecessary features will make the model perform better, but how do we know which $\theta$ to reduce?

**Regularization**

Small values for parameters $\theta_0, \theta_1, \dots , \theta_n$

- "Simpler" hypothesis
- Less prone to overfitting

Housing:

- Features: $x_1, x_2, \dots , x_100$
- Parameters: $\theta_0, \theta_1, \theta_2, \dots , \theta_100$

$$ J(\theta) = \left[ \underbrace{\frac{1}{2m} \sum_{i = 1}^{m}（h_\theta(x^{(i）} - y^{(i)})^2}\_{\text{error term}} + \underbrace{\overbrace{\lambda}^{\text{regularisation parameter}} \overbrace{\sum_{j = 1}^{n} \theta_{j}^{2}}^{\text{excludes }\theta_0}}_{\text{regularisation term}} \right] $$

<img width="249" alt="Screenshot 2568-03-04 at 01 52 29" src="https://github.com/user-attachments/assets/441759e2-d2d8-456f-92cf-49c59b0a6c1f" />

> The line gets smoother and smoother with regularisation term added.

In regularized linear regression, we choose $\theta$ to minimize $J(\theta)$.

Regularisation parameter $\alpha$: like learning rate, is a positive number.

What if $\lambda$ is set to an extremely large value (perhaps far too large for our problem, say $\lambda = 10^{10}$)?

<img width="321" alt="Screenshot 2568-03-04 at 01 51 31" src="https://github.com/user-attachments/assets/2b482d25-ec1b-42c9-bb7e-a9aceddfb3a6" />

<img width="268" alt="Screenshot 2568-03-04 at 02 45 45" src="https://github.com/user-attachments/assets/ae954302-2180-4307-a40c-ff2a959814fa" />

It will horribly underfit.

## Regularised linear regression

$$ J(\theta) = \left[ \frac{1}{2m} \sum_{i = 1}^{m}（h_\theta(x^{(i）} - y^{(i)})^2 + \lambda \sum_{i = 1}^{n} \theta_{j}^{2} \right] $$

$$ \underset{\theta}{\text{min }} J(\theta) $$

**Gradient Descent**

$\text{Repeat } \\{$

$$ \theta_0 := \theta_0 - \alpha \frac{1}{m} \overbrace{\sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_{0}^{(i)}}^{\frac{\partial}{\partial \theta_0} J(\theta)} $$

$$ \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_{j}^{(i)} - \frac{\lambda}{m} \theta_j $$

$\\}$

$$ \text{Rearranged }: \theta_j := \theta_j \left( 1 - \alpha \frac{\lambda}{m} \right) - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_{j}^{(i)} $$

**Normal Equation**

```math
X = \begin{bmatrix} (x^{(1)})^{\top} \\ \cdot \\ \cdot \\ \cdot \\ (x^{(m)})^{\top} \end{bmatrix}^{m \times (n + 1)} \qquad y = \begin{bmatrix} y^{(1)} \\ \cdot \\ \cdot \\ \cdot \\ y^{(m)} \end{bmatrix} \in ℝ^m
```

$$ \underset{\theta}{\text{min }} J(\theta) $$

```math
\theta = \left( X^{\top} X + \underbrace{\lambda \begin{bmatrix} 0 &  &  &  &  \\  & 1 &  &  &  \\  &  & 1 &  &  \\  &  &  & \ddots &  \\  &  &  &  & 1  \end{bmatrix}^{(n + 1) \times (n+1)}}_{\text{regularisation term}} \right)^{-1} X^{\top} y
```

```math
\text{e.g., } n = 3 \quad \begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}
```

No need to worry about non-invertibility, with the regularisation term, $X^{\top} X + regularisationTerm $ will alway be inversible.

## Regularised logistic regression

<img width="210" alt="Screenshot 2568-03-04 at 02 02 02" src="https://github.com/user-attachments/assets/de31a9be-8284-4867-99b2-1e49412c87ea" />

$$ h_\theta(x) = g(\theta_0 + theta_1 x_2 + theta_2 x_{1}^{2} + theta_3 x_{1}^{2} x_2 + theta_4 x_{1}^{2} x_{2}^{2} + theta_5 x_{1}^{2} x_{2}^{3} + \dots) $$

Cost function:

$$ J(\theta) = - \left[ \frac{1}{m} \sum_{i=1}^{m} -y^{(i)} \log (h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log (1 - h_{\theta}(x^{(i)})) + \frac{\lambda}{2m} \sum_{j = 1}^{n} \theta_{j}^{2} \right] $$

**Gradient Descent**

$\text{Repeat } \\{$

$$ \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_{0}^{(i)} $$

$$ \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_{j}^{(i)} - \frac{\lambda}{m} \theta_j $$

$\\}$

$$ \text{Rearranged }: \theta_j := \theta_j \left( 1 - \alpha \frac{\lambda}{m} \right) - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_{j}^{(i)} $$

$$ \text{Note} : h_\theta(x) = \frac{1}{1 = e^{-\theta^{\top} x}} $$




