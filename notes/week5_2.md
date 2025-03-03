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

$$ \text{Predict } \underbrace{y = 1}_{x_1 + x_2 \geq 3} \text{ if } \underbrace{-3 + x_1 + x_2 \geq 0}\_{\theta^{\top} x} $$

<img width="605" alt="Screenshot 2568-03-03 at 20 14 10" src="https://github.com/user-attachments/assets/68f4e9de-3992-4104-8ba7-61bccaf6b2bf" />

**Non-linear decistion boundaries**

Polynomial features

$$ h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \theta_3 x_{1}^{2} + \theta_4 x_{2}^{2}) $$

```math
\text{If } \theta = \begin{bmatrix} -1 \\ 0 \\ 0 \\ 1 \\ 1 \end{bmatrix} \quad \text{Then } h_\theta(x) = g(-1 + 0 x_1 + 0 x_2 + 1 x_{1}^{2} + 1 x_{2}^{2})
```
$$ \text{Predict } y = 1 \text{ if } \underbrace{-3 + x_1 + x_2 \geq 0}\_{x_{1}^{2} + x_{2}^{2} \geq 1} $$

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

Linear Regression = $J(\theta) \frac{1}{2m} \frac{1}{2} (h_theta(x^{(i)}) - y^{(i)})^2$
