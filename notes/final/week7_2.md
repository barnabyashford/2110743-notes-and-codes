## Bayesian Learning (1)

## Two Roles for Bayesian Methods

1. Provides practical learning algorithms
	- Well-known algorithms
		- Naive Bayes Learning
  	- Bayesian Belief Network Learning
  - Combine prior knowledge (prior probabilities) with observed data (we normally use the latter)
  - Requires prior probabilities

2. Provide useful conceptual framework
	- Act as a "gold standard" for evaluating other algorithms
	- Provide insights to Occam's Razor

	## Bayes Theorem

We normally know:

```math
P(A \mid B) = \frac{P(B \mid A)P(A)}{P(B)}
```

Let's put it this way:

```math
P(h \mid D) = \frac{P(D \mid h)P(h)}{P(D)}
```

- $P(h)$ = prior probability of hypothesis $h$ (probability before getting a look at the data)
	- calculated by our own belief (e.g., Occam's Razor)
- $P(D)$ = prior probability of training data $D$ (just probability of $D$)
- $P(h \mid D)$ = posterior probability of $h$ given $D$ (AFTER getting a look at the data)
- $P(D \mid h)$ = posterior probability of $D$ given $h$
	- assumed that $h$ is true, how many points of the data is true

## Choosing Hypotheses

```math
P(h \mid D) = \frac{P(D \mid h)P(h)}{P(D)}
```

We want the **most probable hypothesis** given training data: *Maximum a Posteriori* hypothesis $h_{MAP}$

```math
\begin{matrix}
h_{MAP} & = \arg \max_{h \in H} P(h \mid D)\\
  & = \arg \max_{h \in H} \frac{P(D \mid h)P(h)}{P(D)}\\
  & = \arg \max_{h \in H} P(D \mid h)P(h)
\end{matrix}
```
> As we need only the arguement $h$ that maximises the function, we dont really need the constant $P(D)$ as we do not need the absolute value of the function.

We can also say that, we need the $h$ that has high probability (not too complex) and fits the data

If assume $P(h_i) = P(h_j)$ then can further simplify, and choose the *Maximum Likelihood* (ML) hypothesis

```math
h_{ML} = \arg \max_{h_i \in H} P(D \mid h_i)
```
> This is just assuming that every $h_i \in H$ has the same $P(h_i)$ and cares only the fact that it fits the data. Which we now know that we also need the hypotheses that has high prior probability as well.

### Implementation example

Does patient have cancer or not?

<img width="460" alt="Screenshot 2568-04-26 at 19 58 39" src="https://github.com/user-attachments/assets/985d33bb-f668-4c9e-8f03-ae2a7fd87d78" />

find $h_{MAP} = \arg \max_{h \in H} P(h \mid +) = \arg \max_{h \in H} P(+ \mid h) \cdot P(h)$ Where $H = \\{ cancer, \neg cancer \\}$

```math
\begin{matrix}
P(cancer) = 0.008 \quad P(\neg cancer) = 0.992 \\
P(+ \mid cancer) = 0.98 \quad P(- \mid cancer) = 0.02\\
P(+ \mid \neg cancer) = 0.03 \quad P(- \mid \neg cancer) = 0.97
\end{matrix}
```

```math
h = cancer : P(+ \mid cancer) \dot P(cancer) = 0.98 \times 0.008 = 0.00764
```

```math
h = \neg cancer : P(+ \mid \neg cancer) \dot P(\neg cancer) = 0.03 \times 0.992 = 0.02976
```

$h_{MAP} = \neg cancer$. The patient does not have cancer, this might seem a bit weird, but for medical apparatus, one needs more accuracy than 98%. Therefore, not weird if you think about it.

## Basic Formulas for Probabilities

- *Product Rule*:

```math
P(A \land B) = P(A \mid B)P(B) = P(B \mid A)P(A)
```

- *Sum Rule*: 

```math
P(A \lor B) = P(A) + P(B) - P(A \land B)
```

- *Theorem of total probability*: if events $A_1 , \dots , A_n$ are mutually exclusive with $\sum_{i = 1}^{n} P(A_i) = 1$, then
> mutually exclusive: don't occur at the same time

```math
P(B) = \sum_{i = 1}^{n} P(B \mid A_i)P(A_i) = \sum_{i = 1}^{n} P(B, A_i)
```

## Brute Force MAP Hypothesis Learner

1. For each hypothesis $h$ in $H$, calculate the posterior probability

```math
P(h \mid D) = \frac{P(D \mid h)P(h)}{P(D)}
```

2. Output the hypothesis $h_{MAP}$ with the highest posterior probability

```math
h_{MAP} = \arg \max_{h \in H} P(h \mid D)
```
