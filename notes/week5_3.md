# Regularisation

## The problem of overfitting

<img width="573" alt="Screenshot 2568-03-04 at 01 27 10" src="https://github.com/user-attachments/assets/3ce5fa6c-a26d-4996-8242-2574684e27fa" />

Overfitting: If we have too many features, the learned hypothesis may fit the training set very well $\left( J(0) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^{2} \approx 0 \right)$, but
fail to generalize to new examples (predict prices on new examples).

<img width="601" alt="Screenshot 2568-03-04 at 01 29 53" src="https://github.com/user-attachments/assets/0acdadc7-5239-4dbb-8350-aca551bc6f2e" />

<img width="592" alt="Screenshot 2568-03-04 at 01 30 15" src="https://github.com/user-attachments/assets/16be5ad3-ed55-48d6-a025-936c4d161111" />

**Addressing overfitting**:

Options:

1. Reduce number of features.
  - Manually select which features to keep.
  — Model selection algorithm.
2. Regularization.
  - Keep all the features, but reduce magnitude/values of parameters $\theta_j$
  - Works well when we have a lot of features, each of which contributes a bit to predicting $y$.

## Cost Function

**Intuition**

<img width="437" alt="Screenshot 2568-03-04 at 01 33 06" src="https://github.com/user-attachments/assets/67ffc2da-00d2-47a5-98ad-3469ca9aa306" />

Suppose we penalize and make $\theta_3$, $\theta_4$ really small.

$$ \underset{\theta}{min } \frac{1}{2m} \sum_{i = 1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \underbrace{1000\theta_{3}^{2}}\_{\theta_3 \approx 0}+ \underbrace{1000\theta_{4}^{4}}\_{\theta_4 \approx 0} $$

**Regularization**

Small values for parameters $\theta_0, \theta_1, \dots , \theta_n$

- "Simpler" hypothesis
- Less prone to overfitting

Housing:

- Features: $x_1, x_2, \dots , x_100$
- Parameters: $\theta_0, \theta_1, \theta_2, \dots , \theta_100$

$$ J(\theta) = \left[ \frac{1}{2m} \sum_{i = 1}^{m}（h_\theta(x^{(i）} - y^{(i)})^2 + \overbrace{\lambda}^{\text{regularisation parameter}} \underbrace{\sum_{i = 1}^{n} \theta_{j}^{2}}_{\text{excludes }\theta_0} \right] $$

<img width="249" alt="Screenshot 2568-03-04 at 01 52 29" src="https://github.com/user-attachments/assets/441759e2-d2d8-456f-92cf-49c59b0a6c1f" />

In regularized linear regression, we choose $\theta$ to minimize $J(\theta)$.

What if $\lambda$ is set to an extremely large value (perhaps far too large for our problem, say $\lambda = 10^{10}$)?

<img width="321" alt="Screenshot 2568-03-04 at 01 51 31" src="https://github.com/user-attachments/assets/2b482d25-ec1b-42c9-bb7e-a9aceddfb3a6" />

It will horribly underfit.

## Regularised linear regression

$$ J(\theta) = \left[ \frac{1}{2m} \sum_{i = 1}^{m}（h_\theta(x^{(i）} - y^{(i)})^2 + \overbrace{\lambda}^{\text{regularisation parameter}} \underbrace{\sum_{i = 1}^{n} \theta_{j}^{2}}_{\text{excludes }\theta_0} \right] $$

$$ \underset{\theta}{\text{min }} J(\theta) $$

**Gradient Descent**

$\text{Repeat } \\{$

$$ \theta_0 := \theta_0 - \alpha \frac{1}{m} \overbrace{\sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_{0}^{(i)}}^{\frac{\partial}{\partial \theta_0} J(\theta)} $$

$$ \theta_j := \theta_j - \alpha \frac{1}{m} \underbrace{\sum_{i=1}^{m}}\_{(j = 1,2,3,\dots,n)} (h_\theta(x^{(i)}) - y^{(i)}) x_{j}^{(i)} - \frac{\lambda}{m} \theta_j $$

$\\}$

$$ \theta_j := \theta_j \left( 1 - \alpha \frac{\lambda}{m} \right) - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_{j}^{(i)} $$

**Normal Equation**

<img width="482" alt="Screenshot 2568-03-04 at 02 01 30" src="https://github.com/user-attachments/assets/07272221-7181-45ac-9913-5dd4bfb51f4b" />


**Non-invertibility**

<img width="561" alt="Screenshot 2568-03-04 at 02 01 05" src="https://github.com/user-attachments/assets/dac5eaff-b668-457c-b980-efb3fb8b4c11" />

## Regularised logistic regression

<img width="210" alt="Screenshot 2568-03-04 at 02 02 02" src="https://github.com/user-attachments/assets/de31a9be-8284-4867-99b2-1e49412c87ea" />




