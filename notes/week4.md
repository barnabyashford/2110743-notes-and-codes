# Linear Regression with One Variable

## Model Representation

### Example: Housing Prices (Portland, OR)

<img width="335" alt="Screenshot 2568-03-02 at 08 14 26" src="https://github.com/user-attachments/assets/217ec559-c92b-43ff-a0fb-ddcfb2bbc6b7" />

Idea: Find the line that go through the center of the data, such that it has the smallest cumulative distant between each data and the line.

> Where y-axis means Price (in 1000s of dollars) and x-axis means Size (feet<sup>2</sup>)

**Supervised Learning**: Give the "right answer" for each example in the data.

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

> Here the "learning algorithm" is "linear regression"

**How do we represent $h$?**

$$ h_{\theta}(x) = \theta_0 + \theta_1 x \quad (\text{ based on the linear equation } y = mx + b) $$

shorthand: $h(x)$

<img width="342" alt="Screenshot 2568-03-03 at 20 44 14" src="https://github.com/user-attachments/assets/ef93dff0-5dd9-4d38-9a09-61df6a3aaf46" />

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

$$ \underset{\theta_0, \theta_1}{\text{minimize }} \overbrace{\frac{1}{2m}}^{\text{from } \frac{1}{m} \text{ to find average of squared errors}} \sum_{i=1}^{\overbrace{m}^{\text{\\#training example}}} (\overbrace{h_{\theta}(x^{(i)})}^{\text{prediction}} - \overbrace{y^{(i)}}^{\text{true value}})^{2 \leftarrow \text{ such that we take only difference and not its polarity}} $$  
$$ h_{\theta}(x^{(i)}) = \theta_0 + \theta_1 x^{(i)} $$

> the 2 in $\frac{1}{2m}$ is a technical number that happens from adding $\frac{1}{2}$ to the equation so that the equation would be more elegant after optimisation.

$$ J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2 $$  
$$ \underset{\theta_0, \theta_1}{\text{minimize }} J(\theta_0, \theta_1) $$

Cost function: $J(\theta_0, \theta_1)$

> From the term simplifiable as $(InputFeature - TargetValue)^2$, this is called "Squared error function".

## Cost Function intuition

### Cost Function simplified

| Component | Default | Simplified |
| :---: | :---: | :---: |
| **Hypothesis** | $h_{\theta}(x) = \theta_0 + \theta_1 x$ | $h_{\theta}(x) = \theta_1 x$ |
| **Parameter(s)** | $\theta_0, \theta_1$ | $\theta_1$ |
| **Cost Function** | $J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2$ | $J(\theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2$ |
| **Goal** | $\underset{\theta_0, \theta_1}{\text{minimize }} J(\theta_0, \theta_1)$ | $\underset{\theta_1}{\text{minimize }} J(\theta_1)$ |

> The simplified version sets $\theta_0$ to 0, as it represents intersect, so that the axes would intersect at exactly 0, leaving only one parameter to deal with.

<img width="612" alt="Screenshot 2568-03-02 at 08 51 27" src="https://github.com/user-attachments/assets/56be0f10-7acc-4e08-9547-0cd67b476196" />

> Supposed that the red x's in the left graph are our data. If we set $\theta_1 = 1$, this is the result from the formulas. We can see that if we set $\theta_1$, the line from the function would go over all datapoint, and gives 0 error, making $\theta_1 = 1$ the value we're looking for.

<img width="603" alt="Screenshot 2568-03-02 at 08 51 39" src="https://github.com/user-attachments/assets/7cf196f6-b607-44a0-97fc-6210903d7f67" />

> This is the case where we set $\theta_1 = 0.5$. Here is the result.
>
> Note: the cost function calculate function vertically.

<img width="615" alt="Screenshot 2568-03-02 at 08 51 51" src="https://github.com/user-attachments/assets/d9e1956c-1eb5-4498-8615-585d8450e3e1" />

> This is the case where we set $\theta_1 = 0$. Here is the result.
>
> You can see that the cost function give a parabola graph. We call this a convex function, where we can find the lowest point where the error is the lowest.

### Now we find $\theta_0$ as well.

**Hypothesis** : $h_{\theta}(x) = \theta_0 + \theta_1 x$

**Parameters** : $\theta_0, \theta_1$

**Cost Function** : $J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)})-y^{(i)})^2$

**Goal** : $\underset{\theta_0, \theta_1}{\text{minimize }} J(\theta_0, \theta_1)$

<img width="542" alt="Screenshot 2568-03-02 at 18 14 11" src="https://github.com/user-attachments/assets/2e789da0-67c1-4ea7-9629-b4446c96a004" />

> Here we set $\theta_0 = 50$ and $\theta_1 = 0.06$.

<img width="489" alt="Screenshot 2568-03-02 at 18 14 28" src="https://github.com/user-attachments/assets/0f5990e5-2ea2-4751-8276-b9e572dcf2a4" />

> Earlier there was only one parameter to tune, so the graph was 2D. Here we have more dimension $\theta_0$, now the graph is 3D.
>
> This case, the cost function is convex, showing that we can really find the lowest point that will in turn give us the best $\theta_0, \theta_1$

<img width="541" alt="Screenshot 2568-03-02 at 18 14 41" src="https://github.com/user-attachments/assets/cd0a3901-446b-4464-aee0-2c18b3fe93a8" />

> The graph on the right is called "contour graph", where while the graph is in 2D, it still gives us the information of $\theta_0, \theta_1$ and $J(\theta_0, \theta_1)$ .
>
> Note: the colour of the line signifies value of $J(\theta_0, \theta_1)$, ranging from blue (low) to red (high).

<img width="542" alt="Screenshot 2568-03-02 at 18 14 53" src="https://github.com/user-attachments/assets/395905c0-8ea0-418d-8a09-d3b8a02efaec" />

<img width="522" alt="Screenshot 2568-03-02 at 18 15 02" src="https://github.com/user-attachments/assets/6ab8c4a1-7b2f-4206-95cd-3e61f9278caf" />

<img width="551" alt="Screenshot 2568-03-02 at 18 15 11" src="https://github.com/user-attachments/assets/a25a32e3-3f4d-478b-a8b3-75d75d8e34c2" />

## Gradient Descent

A function: $J(\theta_0, \theta_1)$ (Use of GD isn't limited to univariate linear regression)

Goal: $\underset{\theta_0, \theta_1}{\text{min }} J(\theta_0, \theta_1)$

**Outline**:

- Start with some $\theta_0, \theta_1$ (say $\theta_0 = 0$, $\theta_1 = 0$)
- Keep changing $\theta_0, \theta_1$ to reduce $J(\theta_0, \theta_1)$ until we hopefully end up at a minimum

<img width="460" alt="Screenshot 2568-03-02 at 18 17 54" src="https://github.com/user-attachments/assets/9df22fa9-aec5-4646-b014-bb72d603b04d" />

> Before, we dealt with $J(\theta_0, \theta_1)$ which was a convex function. Although in some case, it might not be a convex function.

<img width="519" alt="Screenshot 2568-03-02 at 18 18 04" src="https://github.com/user-attachments/assets/e8cfb940-c6b4-41e1-956a-f563d593f586" />

> In this case, the $J(\theta_0, \theta_1)$ function is not convex, so whether we will end at "global minimum" or not, it all depends on the initial value of the parameters.
>
> We don't have to worry about this, as Linear Regression's $J(\theta_0, \theta_1)$ will normally be convex.


### Gradient Descent Algorithm

$\text{repeat until convergence \(parameters no longer change\) }\\{$  

$$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1) \quad (\text{for }j=0\text{ and }j=1) $$

$\\}$

| Assignment ($:=$) | Truth assertion ($=$) |
| :---: | :---: |
| $a := b$ | $a = b$ |
| $a := a+1$ | $a = a+1$ (wrong) |

Explanation:

- $\alpha$ &larr; learning rate
- $"(\text{for }j=0\text{ and }j=1)"$ means that in each step, we simulatneously update $\theta_0$ and $\theta_1$

| Step | Correct: Simultaneous update | Incorrect: |
| --- | --- | --- |
| 1 | $\text{temp0 }:=\theta_0 - \alpha \frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1)$ | $\text{temp0 }:=\theta_0 - \alpha \frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1)$ |
| 2 | $\text{temp1 }:=\theta_1 - \alpha \frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1)$ | $\theta_0:=\text{temp0}$ |
| 3 | $\theta_0:=\text{temp0}$ | $\text{temp1 }:=\theta_1 - \alpha \frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1)$ |
| 4 | $\theta_1:=\text{temp1}$ | $\theta_1:=\text{temp1}$ |
> calculate and store the updated value for each parameter $\text{tempj} first, do not update one by one.

## Gradient Descent Intuition

**Gradient Descent Algorithm**

$\text{repeat until convergence }\\{$  

$$ \theta_j := \theta_j - \underbrace{\alpha}_{\text{learning rate}} \underbrace{\frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1)}\_{\text{derivative}} \quad (\text{simultaneously update }j=0\text{ and }j=1) $$

$\\}$

- $\underset{\theta_1}{\text{min }} J(\theta_1) \quad \theta_1 \in ℝ$

<img width="532" alt="Screenshot 2568-03-03 at 00 01 00" src="https://github.com/user-attachments/assets/dea55bbc-6947-4040-8c20-fa867cbe2a64" />

> $\alpha$ must be a small positive number.
>
> If the derivative is a positive number, $\theta_1$ decreases, going to the left side of the graph.
> And if it's the opposite, the opposite happens.

$$ \theta_1 := \theta_1 - \alpha \frac{\partial}{\partial \theta_1} J(\theta_1) $$

If $\alpha$ is too small, gradient descent can be slow, which is not a big problem, just time consuming.

<img width="222" alt="Screenshot 2568-03-03 at 00 11 03" src="https://github.com/user-attachments/assets/02b525c4-4208-4ca4-9963-25ee950e92b6" />

If $\alpha$ is too large, gradient descent can overshoot the minimum. It may fail to converge, or even diverge. This is bad, as we will never get the optimal parameter.

<img width="217" alt="Screenshot 2568-03-03 at 00 11 53" src="https://github.com/user-attachments/assets/f5b4c2f9-6e65-43e8-88e3-35f526d660eb" />

<img width="491" alt="Screenshot 2568-03-03 at 00 12 10" src="https://github.com/user-attachments/assets/460f9f0c-cc10-4346-977b-d9fd34256adc" />

> This image shows the idea of changing the learning rate as we approach the global minimum to not overshoot. But we don't really need to do that.

Gradient descent can converge to a local minimum, even with the learning rate $\alpha$ fixed. 

$$ \theta_1 := \theta_1 - \alpha \frac{d}{d \theta_1} J(\theta_1) $$

To fix this, as we approach a local minimum, gradient descent will automatically take smaller steps. So, no need to decrease $\alpha$ over time.

<img width="308" alt="Screenshot 2568-03-03 at 21 40 04" src="https://github.com/user-attachments/assets/59f64977-0fe5-460a-905e-44fa372e7a88" />

## Gradient descent for linear regression

| Gradient Descent Algorithm | Linear Regression Model |
| :---: | :---: |
| $\text{repeat until convergence }\\{$<br>$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1) \quad (\text{for }j=0\text{ and }j=1)$<br>$\\}$ | $h_\theta(x) = \theta_0 + \theta_1 x$<br>$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$ |

$$ \frac{\partial}{\partial \theta_j} J(\theta_0, \theta_1) = \frac{\partial}{\partial \theta_j} \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 $$

$$ = \frac{\partial}{\partial \theta_j} \frac{1}{2m} \sum_{i=1}^{m} (\theta_0 + \theta_1 x^{(i)} - y^{(i)})^2 $$

$$ j = 0 : \frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) $$

$$ j = 1 : \frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1) = \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)} $$

**Gradient Descent Algorithm**

$\text{repeat until convergence }\\{$  

$$ \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \quad \left( \text{recap: } \left[ \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \right] \leftarrow \left[ \frac{\partial}{\partial \theta_0} J(\theta_0, \theta_1) \right] \right) $$

$$ \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)} \quad \left( \text{recap: } \left[ \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x^{(i)} \right] \leftarrow \left[ \frac{\partial}{\partial \theta_1} J(\theta_0, \theta_1) \right] \right) $$

$\\}$

- update $\theta_0$ and $\theta_1$ simultaneously

<img width="533" alt="Screenshot 2568-03-03 at 00 35 16" src="https://github.com/user-attachments/assets/f1831fe6-5c96-41c1-8055-81cda52f41d3" />

<img width="519" alt="Screenshot 2568-03-03 at 00 35 26" src="https://github.com/user-attachments/assets/c3f610a0-74a2-433b-ab10-0ff4333088c6" />

<img width="540" alt="Screenshot 2568-03-03 at 00 35 36" src="https://github.com/user-attachments/assets/08a11173-63b2-4a69-afb5-3580fc6a43cf" />

<img width="551" alt="Screenshot 2568-03-03 at 00 35 54" src="https://github.com/user-attachments/assets/9173c984-0e4e-495c-b648-e4775a271edd" />

<img width="565" alt="Screenshot 2568-03-03 at 00 36 03" src="https://github.com/user-attachments/assets/7dba1d8f-960b-4cfc-a270-1cf6d3d563bc" />

<img width="548" alt="Screenshot 2568-03-03 at 00 36 11" src="https://github.com/user-attachments/assets/1a273b50-3af5-4f71-b8d3-d27d857a2493" />

<img width="546" alt="Screenshot 2568-03-03 at 00 36 19" src="https://github.com/user-attachments/assets/57a9b87f-b3c0-4fad-b282-88dbc0df48f7" />

<img width="544" alt="Screenshot 2568-03-03 at 00 37 14" src="https://github.com/user-attachments/assets/4367a554-ddff-481b-9755-bdf76f850ff4" />

<img width="536" alt="Screenshot 2568-03-03 at 00 37 42" src="https://github.com/user-attachments/assets/cfa885ab-8936-48a6-8b75-2920da952540" />

<img width="551" alt="Screenshot 2568-03-03 at 00 37 50" src="https://github.com/user-attachments/assets/1dc9c71a-4c86-4d7d-a08f-61f081f0425a" />

<img width="522" alt="Screenshot 2568-03-03 at 00 38 00" src="https://github.com/user-attachments/assets/a98edeaa-b149-4327-9e95-7d5901ac0ca2" />

<img width="557" alt="Screenshot 2568-03-03 at 00 38 09" src="https://github.com/user-attachments/assets/4103096f-41c9-421d-9d96-aa46bf7ada15" />

**"Batch" Gradient Descent**

$$ \sum_{i=1}^{m} (h_\theta(x^{(i)} - y^{(i)}) $$

"Batch": Each step of gradient descent uses all the training examples.
