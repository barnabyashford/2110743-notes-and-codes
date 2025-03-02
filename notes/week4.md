# Linear Regression with One Variable

## Model Representation

### Example: Housing Prices (Portland, OR)

<img width="335" alt="Screenshot 2568-03-02 at 08 14 26" src="https://github.com/user-attachments/assets/217ec559-c92b-43ff-a0fb-ddcfb2bbc6b7" />

> Where y-axis means Price (in 1000s of dollars) and x-axis means Size (feet<sup>2</sup>)

**Supervised Learning**: Given the "right answer" for each example in the data.

**Regression Problem**: Predict real-valued output

> **Classification Problem**: Predict discreet-valued output

#### Training set of housing prices (Portland, OR)

| Size in feet<sup>2</sup> ($x$) | Price (\$) in 1000's ($y$) |
| :---: | :---: |
| 2104 | 460 |
| 1416 | 232 |
| 1534 | 315 |
| 852 | 178 |
| ... | ... |

**Notation**:

- $m$ = Number of training examples ($m = 47$)
- $x$'s = "input" variable/features
- $y$'s = "output" variable/"target" variable

$(x,y)$ - one training example  
$(x^{(i)},y^{(i)})$ - $i^{th}$ training example

$x^{(1)} = 2104$  
$x^{(2)} = 1416$  
$y^{(1)} = 460$

<img width="314" alt="Screenshot 2568-03-02 at 08 25 27" src="https://github.com/user-attachments/assets/97bf4042-189a-4e6d-9c98-661df8e56b3e" />

**How do we represent $h$?**

$$ h_{\theta}(x) = \theta_0 + \theta_1 x $$

shorthand: $h(x)$

This particular type of Lineal Regression model is called: "Lineal Regression with one variable" or "Univariate Lineal Regression".

## Cost Function

**Training set ($m = 47$)**:

| Size in feet<sup>2</sup> ($x$) | Price (\$) in 1000's ($y$) |
| :---: | :---: |
| 2104 | 460 |
| 1416 | 232 |
| 1534 | 315 |
| 852 | 178 |
| ... | ... |

Hypothesis: $h_{\theta}(x) = \theta_0 + \theta_1 x$

$\theta_i$'s: Parameters

How to choose $\theta_i$'s?

<img width="568" alt="Screenshot 2568-03-02 at 08 30 29" src="https://github.com/user-attachments/assets/d9b34162-7b69-4cdf-837b-52c6a845a41a" />

<img width="269" alt="Screenshot 2568-03-02 at 08 30 59" src="https://github.com/user-attachments/assets/85fe4721-8f08-4e90-b14e-e6ece20a8bdc" />

Idea: Choose $\theta_0, \theta_1$ so that $h_{\theta}(x)$ is close to $y$ for our training example $(x,y)$

$$ \underset{\theta_0, \theta_1}{\text{minimize }} \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2 $$  
$$ h_{\theta}(x^{(i)}) = \theta_0 + \theta_1 x^{(i)} $$

$$ J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2 $$  
$$ \underset{\theta_0, \theta_1}{\text{minimize }} J(\theta_0, \theta_1) $$

Cost function: $J(\theta_0, \theta_1)$

> From the term simplifiable as $(InputFeature - TargetValue)^2$, this is called "Squared error function".

## Cost Function intuition

### Part I

| Component | Default | Simplified |
| :---: | :---: | :---: |
| **Hypothesis** | $h_{\theta}(x) = \theta_0 + \theta_1 x$ | $h_{\theta}(x) = \theta_1 x$ |
| **Parameter(s)** | $\theta_0, \theta_1$ | $\theta_1$ |
| **Cost Function** | $J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2$ | $J(\theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2$ |
| **Goal** | $\underset{\theta_0, \theta_1}{\text{minimize }} J(\theta_0, \theta_1)$ | $\underset{\theta_1}{\text{minimize }} J(\theta_1)$ |
> The simplified version sets $\theta_0$ to 0, as it represents intersect, so that the axes would intersect at exactly 0, making the line cleaner to look at.

<img width="612" alt="Screenshot 2568-03-02 at 08 51 27" src="https://github.com/user-attachments/assets/56be0f10-7acc-4e08-9547-0cd67b476196" />

<img width="603" alt="Screenshot 2568-03-02 at 08 51 39" src="https://github.com/user-attachments/assets/7cf196f6-b607-44a0-97fc-6210903d7f67" />

<img width="615" alt="Screenshot 2568-03-02 at 08 51 51" src="https://github.com/user-attachments/assets/d9e1956c-1eb5-4498-8615-585d8450e3e1" />

## Part II

**Hypothesis** : $h_{\theta}(x) = \theta_0 + \theta_1 x$

**Parameters** : $\theta_0, \theta_1$

**Cost Function** : $J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2$

**Goal** : $\underset{\theta_0, \theta_1}{\text{minimize }} J(\theta_0, \theta_1)$

<img width="542" alt="Screenshot 2568-03-02 at 18 14 11" src="https://github.com/user-attachments/assets/2e789da0-67c1-4ea7-9629-b4446c96a004" />

<img width="489" alt="Screenshot 2568-03-02 at 18 14 28" src="https://github.com/user-attachments/assets/0f5990e5-2ea2-4751-8276-b9e572dcf2a4" />

<img width="541" alt="Screenshot 2568-03-02 at 18 14 41" src="https://github.com/user-attachments/assets/cd0a3901-446b-4464-aee0-2c18b3fe93a8" />

<img width="542" alt="Screenshot 2568-03-02 at 18 14 53" src="https://github.com/user-attachments/assets/395905c0-8ea0-418d-8a09-d3b8a02efaec" />

<img width="522" alt="Screenshot 2568-03-02 at 18 15 02" src="https://github.com/user-attachments/assets/6ab8c4a1-7b2f-4206-95cd-3e61f9278caf" />

<img width="551" alt="Screenshot 2568-03-02 at 18 15 11" src="https://github.com/user-attachments/assets/a25a32e3-3f4d-478b-a8b3-75d75d8e34c2" />

## Gradient Descent

A function: $J(\theta_0, \theta_1)$  
Goal: $\underset{\theta_0, \theta_1}{\text{min }} J(\theta_0, \theta_1)$

**Outline**:

- Start with some $\theta_0, \theta_1$
- Keep changing $\theta_0, \theta_1$ to reduce $J(\theta_0, \theta_1)$ until we hopefully end up at a minimum

<img width="460" alt="Screenshot 2568-03-02 at 18 17 54" src="https://github.com/user-attachments/assets/9df22fa9-aec5-4646-b014-bb72d603b04d" />

<img width="519" alt="Screenshot 2568-03-02 at 18 18 04" src="https://github.com/user-attachments/assets/e8cfb940-c6b4-41e1-956a-f563d593f586" />

### Gradient Descent Algorithm

$\text{repeat until convergence }\\{$  
$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} (\text{for }j=0\text{ and }j=1) $$  
$\\}$






