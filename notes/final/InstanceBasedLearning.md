# Instance Based Learning

- $k$-Nearest Neighbour
- Locally weighted regression
- Radial basis functions
- Case-based reasoning
- Lazy and eager learning

## Instance Based Learning

Key idea: store all training data as $\langle x_i,f(x_i) \rangle$

- Nearest neighbour:
 - Given a query instance $x_q$, find the nearest training example (in Euclidean space) $x_n$ then estmate
 
```math
\hat{f}(x_q) \leftarrow f(x_n)
```

- $k$-Nearest Neighbour:
 - Find $k$ nearest training example(s), then use majority vote to estimate $x_q$

```math
\hat{f}(x_q) \leftarrow \frac{\sum_{i=1}^{k} f(x_i)}{k}
```

 - Example:  
   Given this data

   | $x$ | $f(x)$ |
   | --- | --- |
   | 1 | 3.8 |
   | 3 | 10.2 |
   | 7 | 22 |
   | 10 | 20 |

   Set $k = 3$ and $x_q = 2$

   <img width="405" alt="Screenshot 2568-04-28 at 19 43 36" src="https://github.com/user-attachments/assets/f6366305-de88-4cf5-ae48-9d061a00908e" />

```math
\hat{f}(2) = \frac{3.8 + 10.2 + 22}{3} = 12
```

## When to Consider Nearest Neighbour

- Intstances map to points in $\Re^n$ (numeric, if instances are symbolic, they need to be featurised first)
- $\leq 20$ attributes
- Lots of training data

Advantages:

- Fast training
- Learn complex target functions
- Don't lose information (some learning algorithm discard non-discriminating attributes)

Disadvantages:

- Slow at query
- Maintaining irrelevant attributes can make the learner easily tricked by them.

## Voronoi Diagram

Learning algorithms like KNN can be visualised using Voronoi Diagram

Let's talk normal Venn Diagram first:

<img width="214" alt="Screenshot 2568-04-28 at 19 53 12" src="https://github.com/user-attachments/assets/367f4a27-e20a-413e-9593-3f98fc0c98a9" />

> If set $k=1$ the $x_q$ is a positive class, but if set $k=5$ negative wins the vote.

Using Voronoi Diagram, we can visualise the case of $k=1$ as like this:

<img width="425" alt="Screenshot 2568-04-28 at 19 52 15" src="https://github.com/user-attachments/assets/45ead744-673d-4384-9206-01e4c97871f0" />

## Behaviour in the Limit

Consider $p(x)$ defines probability that instance $x$ will be labelled 1 (positive) versus 0 (negative).

Nearest Neighbour ($k=1$):

- The bigger the training data is, the nearer it approaches Gibbs Algorithm  
  Gibbs: with probability $p(x)$ predict 1, else 0

  > Gibbs Algorithm: from a hypothesis space $H$, randomly pick a hypothesis $h$.
  > 
  > In this case: A query instance $x_q$ is blindly put in the space and the nearest neighbour was chosen, each size of dataset will return different answer.

$k$-Nearest Neighbour:

- The bigger the training data is, and the bigger $k$ is, the nearer it approaches Bayes Optimal  
  Bayes Optimal: if $p(x) > .5$ then predict 1, else 0

  > Bayes Optimal: using many datapoints to help answering the question.
  > 
  > In this case: Every nearest $k$ neighbours helps answering the question

***Note :*** Gibbs algorithm's error is twice the expected error of Bayes optimal

## Distance-Weighted $k$NN

Idea: Using distance to tune weights for each datapoint as compared to $x_q$

Want to weight nearer neighbour more heavily:

```math
\hat{f}(x_q) \leftarrow \frac{\sum_{i=1}^{k} w_i f(x_i)}{\sum_{i=1}^{k} w_i}
```

where:

```math
w_i \equiv \frac{1}{d(x_q, x_i)^2}
```

Therefore, the case earlier:

| $x$ | $f(x)$ |
| --- | --- |
| 1 | 3.8 |
| 3 | 10.2 |
| 7 | 22 |
| 10 | 20 |

$k = 3$, $x_q = 2$

```math
\hat{f}(2) = \frac{3.8 \times \frac{1}{1} + 10.2 \times \frac{1}{1} + 22 \times \frac{1}{25}}{\frac{1}{1} + \frac{1}{1} + \frac{1}{25}} = 7.3
```

> See that now the result gets more reasonable, as earlier the result seems weird for 2 to get 12 while 1 get 3.8 and 3 get 10.2.

## Curse of Dimensionality

Imagine instances described by 20 attributes, but only 2 are relevant

*Curse of Dimensionality:* nearest neighbour is easily mislead when high-dimensional

One approach:
- Stretch $j$th axis by weight $z_j$, where $z_1, \dots, z_n$ chosen to minimise prediction error &rarr; the idea is the let the relevant axis measures closer while non-relevant gets further and harder to affect the decision.
- To choose $z_n$: Use cross-validation to automatically choose weight $z_1, \dots, z_n$.
- Setting $z_j$ to zero to eliminate $j$th irrelevant dimension.

## Locally Weighted Regression

$k$NN forms local approximation to $f$ for each query point $x_q$. Here we form a "model"

- Fit a linear function to $k$ nearest neighbour &rarr; e.g., find $k$ nearest points and fit linear regression model
- or even quadratic, ...
- Produces "piecewise approximation" to $f$ &rarr; approximation of $k$ nearest neighbours

Errors to minimise:

- SE over $k$ neighbours

```math
E_1(x_q) \equiv \frac{1}{2} \sum_{x \in \text{nearest neighbours of }x_q} (f(x) - \hat{f}(x))^2
```

- Distance-weighted SE over all neighbours

```math
E_2(x_q) \equiv \sum_{x \in D} (f(x) - \hat{f}(x))^2 K(d(x_q, x))
```

## Radial Basis Function (RBF) Networks

- Global approximation from linear combination of local approximations
- Used in many implementation e.g., Image Classification
- A different kind of neural network
- Closely related to distance-weighted regression, but is a "eager" learning while most instance based learning algorithms are "lazy"

<img width="484" alt="Screenshot 2568-04-28 at 21 09 46" src="https://github.com/user-attachments/assets/620183a2-2188-4e17-8140-176d88c2cc6f" />

> Note that there isn't any weight applied to inputs yet. Each hidden nodes represents each example $ex_k$, each input will be measured for their distance between themselves and examples, then the result are linear combination of all (with $w_0$ of course).

Where $a_i(x)$ are the attributes describing instance $x$, and

```math
f(x) = w_0 + \sum_{u = 1}^{k} w_u K_u (d(x_u, x))
```

One common choice for $K_u(d(x_u, x))$ is

```math
K_u(d(x_u, x)) = e^{\frac{1}{2 \sigma_{u}^{2}} d^{2}(x_u,x)}
```

Note: $x_u$ is the centre of normal distribution graph (as seen in the image)

### Training RBF Networks

Q1: What $x_u$ to use for each kernel function $K_u(d(x_u,x))$ (Where do you get them?)

- Training instances? &rarr; reflect actual distribution, high cost, there's too many
- Scatter uniformly throughout the instance space &rarr; randomly pick from the instace space and hope that the chosen data reflects the actual distribution

Q2: How do we train weights? (normally use Gaussian $K_u$)

- First choose variance (and perhaps mean) for each $K_u$ &rarr; initialisaion perhaps?
 - e.g., use EM
- Then hold $K_u$ fixed, and train linear output layer
 - efficient methods to fit linear funciton

## Case-Based Reasoning

Specifically designed for symbolic data, applicable with algorithms mentioned earlier

## Lazy and Eager Learning

**Lazy** &rarr; wait for query before generalising

**Eager** &rarr; generalise before seeing query

Does it matter?

- Eager learner try to find global approximation
- While lazy learner create local approximation (depends on the query)
- If use the same $H$, lazy can represent a more complex functions (many query-focused approximation).
