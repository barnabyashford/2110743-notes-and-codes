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

If we use pure linear function like the one we use in Linear Regression, it will output put a continuous value.

To classify the data, we might set a threshold for the output say:

Threshold classifier output $h_\theta(x)$ at o.5:
- If $h_\theta(x) \geq 0.5$, predict "y=1"
- If $h_\theta(x) < 0.5$, predict "y=0"

This is not ideal, because the model will still think of the output and the label as continuous values and will fit to data accordongly, if we plot the linear function onto the graph, we will get a straight line, and give the model another instance to train on and plot the graph again, the straight line will change drastically.

Classification: $y = 0 \text{ or } 1$  

If we stil treat the label as continuous values, $h_theta(x)$ can range even wider than the labels. So, we want $h_\theta(x)$ to only be be $>1$ or $<0$ like $y$.

$$ \underset{\text{the task is classfication (predict discreet value), not regression (predict continuous value)}}{\text{Logistic Regression}}: 0 \leq h_\theta(x) \leq 1 $$

## Hypothesis representation

**Logistic Regression Model**

Find $0 \leq h_\theta(x) \leq 1$

linear regression: $h_\theta(x) = \theta^{\top} x$ (output ranges $(-\infty, \infty)$ )

How do we keep $h_\theta(x)$ between 0 and 1? We use "Logistic Function" (Here we use "Sigmoid Function" $g(z)$ ).

$$ g(z) = \frac{1}{1 + e^{-z}} $$

$$ h_\theta(x) = g(\theta^{\top} x) = \frac{1}{1 + e^{-\theta^{\top} x}} $$

<img width="312" alt="Screenshot 2568-03-03 at 19 26 36" src="https://github.com/user-attachments/assets/82a68289-1182-47db-b503-25b338764413" />

**Interpretation of Hypothesis Output**

$h_\theta(x)$ = estimated probability that $y = 1$ on input x

Example: If

```math
x = \begin{bmatrix} x_0 \\ x_1 \end{bmatrix} = \begin{bmatrix} 1 \\ \text{tumourSize} \end{bmatrix}
```

$$ h_\theta(x) = 0.7 $$

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

- predict " $y = 1$ if $h_\theta(x) \geq 0.5$ "
  - $g(z) \geq 0.5 \text{ when } z \geq 0$
  - $\theta^{\top} x \geq 0$

- predict " $y = 0$ if $h_\theta(x) < 0.5$ "
  - $g(z) < 0.5 \text{ when } z < 0$
  - $\theta^{\top} x < 0$

**Decision Boundary**

$$ h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2) $$

```math
\text{If } \theta = \begin{bmatrix} -3 \\ 1 \\ 1 \end{bmatrix} \quad \text{Then } h_\theta(x) = g(-3 + 1 x_1 + 1 x_2)
```

$$ \text{Predict } y = 1 \text{ if } -3 + x_1 + x_2 \geq 0 $$

<img width="605" alt="Screenshot 2568-03-03 at 20 14 10" src="https://github.com/user-attachments/assets/68f4e9de-3992-4104-8ba7-61bccaf6b2bf" />

The decision boundary is the line that we got from training the model, and we can see that it lies between the classes, separating them.

**Non-linear decistion boundaries**

Polynomial features

$$ h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_{1}^{2} + \theta_4 x_{2}^{2}) $$

```math
\text{If } \theta = \begin{bmatrix} -1 \\ 0 \\ 0 \\ 1 \\ 1 \end{bmatrix} \quad \text{Then } h_\theta(x) = g(-1 + 0 x_1 + 0 x_2 + 1 x_{1}^{2} + 1 x_{2}^{2})
```
$$ \text{Predict } y = 1 \text{ if } -1 + x_{1}^{2} + x_{2}^{2} \geq 0 $$

<img width="217" alt="Screenshot 2568-03-03 at 20 28 52" src="https://github.com/user-attachments/assets/3341755d-b03f-4109-bf5c-6f5aa2e4b367" />

It can expand as far as we want it to.

$$ h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_{1}^{2} + \theta_4 x_{1}^{2} x_2 + \theta_5 x_{1}^{2} x_{2}^{2} + \theta_6 x_{1}^{3} x_2 + \cdots) $$

<img width="175" alt="Screenshot 2568-03-03 at 20 29 05" src="https://github.com/user-attachments/assets/465a976d-1457-4ccf-8c71-7c84acb03bb0" />

## Cost function

Training set $\\{ (x^{(1)}, y^{(1)}), (x^{(2)}, y^{(2)}), \cdots, (x^{(m)}, y^{(m)}) \\}$

$m$ examples

```math
x \in \begin{bmatrix} x_0 \\ x_1 \\ \cdots \\ x_n \end{bmatrix} \in ℝ^{n + 1} \quad x_0 = 1, y \in \{ 0,1 \}
```

$$ h_\theta(x) = \frac{1}{1-e^{-\theta^{\top} x}} $$

How do we choose parameter $\theta$ ?

**Cost function**

Linear Regression: $J(\theta) = \frac{1}{m} \sum_{i = 1}^{m} \frac{1}{2} (h_\theta(x^{(i)}) - y^{(i)})^2$

Logistic Regression: $J(\theta) = \frac{1}{m} \sum_{i = 1}^{m} \text{Cost}(h_\theta(x), y))$

$$ \text{Cost}(h_\theta(x), y) = \frac{1}{2} (\overbrace{(h_\theta(x)}^{\frac{1}{1 + e^{-\theta^{\top} x}}} - y))^2 $$

But this (squared error approach) doesn't work well, because if we map this to a graph, it will output a non-convex graph, something we do not want.

<img width="249" alt="Screenshot 2568-03-04 at 00 23 23" src="https://github.com/user-attachments/assets/fd8dc89f-94c1-45dd-a9f6-d7adeebc4727" />

Here's what we want:

<img width="282" alt="Screenshot 2568-03-04 at 00 23 44" src="https://github.com/user-attachments/assets/902af628-3d9b-4a31-8b60-e71c286da562" />

### Logistic Regression cost function

```math
\text{Cost}(h_{\theta}(x), y) = \begin{cases} -\log(h_{\theta}(x)) & \text{if } y = 1 \\ -\log(1 - h_{\theta}(x)) & \text{if } y = 0 \end{cases}
```

Why this works:

As $0 \leq h_\theta(x) \leq 1$, its $-\log$ is fixed between zero and $\infty$.

$\text{Cost} = 0 \text{ if } y = 1, h_\theta(x) = 1$  
$\text{But as } h_\theta(x) = 1 \leftarrow \text{Cost} = \infty$

Here we have found a way to penalise if the model makes incorrect prediction as the further the prediction goes away from the true label, the larger the penalty.

<img width="241" alt="Screenshot 2568-03-04 at 00 26 50" src="https://github.com/user-attachments/assets/0074619c-7997-433c-8785-581a0ca591e2" />

On the other hand:

<img width="476" alt="Screenshot 2568-03-04 at 00 33 26" src="https://github.com/user-attachments/assets/106b9a5e-0e6b-432c-a6bf-b697ccbb7d27" />

Again, this gives us a way to penalise if the model makes incorrect prediction as the further the prediction goes away from the true label, the larger the penalty.

The cost function is now convex as well.

## Simplified cost function and gradient descent

**Logistic Regression cost function**

$$ J(\theta) = \frac{1}{m} \sum_{i = 1}^{m} \text{Cost}(h_\theta(x^{(i)}), y^{(i)}) $$

```math
\text{Cost}(h_{\theta}(x), y) = \begin{cases} -\log(h_{\theta}(x)) & \text{if } y = 1 \\ -\log(1 - h_{\theta}(x)) & \text{if } y = 0 \end{cases}
```

This is not an elegant way according to mathematician. So we simpliefies it into a simpler version. As $y$ is either 0 or 1:

$$ \text{Cost}(h_\theta(x), y) = \underbrace{-y}\_{= 0 \text{ if } y = 0} \log(h_{\theta}(x)) - (\underbrace{(1 - y)}\_{= 0 \text{ if } y = 1} \log(1 - h_{\theta}(x))) $$

$$ \text{If } y = 1 : \text{Cost}(h_\theta(x), y) = -\log(h_{\theta}(x)) $$

$$ \text{If } y = 0 : \text{Cost}(h_\theta(x), y) = -\log(1 - h_{\theta}(x)) $$

Now **Logistic Regression cost function** is:

$$ J(\theta) = \frac{1}{m} \sum_{i = 1}^{m} \text{Cost}(h_\theta(x^{(i)}), y^{(i)}) $$

$$ = - \frac{1}{m} \left[ \sum_{i=1}^{m} -y^{(i)} \log (h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log (1 - h_{\theta}(x^{(i)})) \right] $$

To fit parameters $\theta$ :

$$ \underset{\theta}{\text{min }} J(\theta) $$

To make a prediction given new $x$ :

$$ h_\theta(x) = \frac{1}{1 + e^{-\theta^{\top} x}} $$

**Gradient Descent**

$$ J(\theta) = - \frac{1}{m} \left[ \sum_{i=1}^{m} -y^{(i)} \log (h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log (1 - h_{\theta}(x^{(i)})) \right] $$

Want $\underset{\theta}{\text{min }} J(\theta)$

$\text{Repeat } \\{$

$$ \theta_j := \theta_j = \alpha \overbrace{\frac{\partial}{\partial \theta_j} J(\theta)}^{\sum_{i = 1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) x_{j}^{(i)}} \quad (\text{simultaneously update all } \theta_j) $$

$$ \\} $$

## Advanced Optimisation

Cost function $J(\theta)$. Want $\underset{\theta}{min } J(\theta)$.

Given $(\theta)$, we have code that can compute: 

- $J(\theta)$  
- $\frac{\partial}{\partial \theta_j} J(\theta)$  (for $j = 0, 1, \dots, n$ )

Gradient Descent:

$\text{Repeat} \\{$

$$ \theta_j := \theta_j = \alpha \frac{\partial}{\partial \theta_j} J(\theta) $$

$\\}$

**Optimization algorithm**

Given $\theta$, we have code that can compute:

- $J(\theta)$  
- $\frac{\partial}{\partial \theta_j} J(\theta)$  (for $j = 0, 1, \dots, n$ )

Optimization algorithms:

- Gradient descent
- Conjugate gradient
- BFGS
- L-BFGS

Advantages:

- No need to manually pick $\alpha$
- Often faster than gradient descent.

Disadvantages:

- More complex

## Multi-class classification: One-vs-all

**Multiclass classification examples**

Email foldering/tagging: Work ($x_1$), Friends ($x_2$), Family ($x_3$), Hobby ($x_4$)

Medical diagrams: Not ill ($x_1$), Cold, ($x_2$) Flu ($x_3$)

Weather: Sunny ($x_1$), Cloudy ($x_2$), Rain ($x_3$), Snow ($x_4$)

<img width="539" alt="Screenshot 2568-03-04 at 01 17 12" src="https://github.com/user-attachments/assets/c738e45f-beed-4597-8d37-c4e9cf005787" />

<img width="498" alt="Screenshot 2568-03-04 at 01 18 22" src="https://github.com/user-attachments/assets/140fc04b-7534-48eb-9e96-967c92c703d6" />

- Train a logistic regression classifier $h_\theta(x)$ for each class $i$ to predict the probability that $y = i$.
- On a new input $x$, to make a prediction, pick the class $i$ that maximizes

$$ \underset{i}{max } h_{\theta}^{(i)} (x) $$
