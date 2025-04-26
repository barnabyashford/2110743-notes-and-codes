# Evaluating Hypotheses

## Two Definitions of Error

The **true error** of hypothesis $h$ with respect to target function $f$ and distribution $D$ is the probability that $h$ will misclassify an instance drawn at random according to $D$. 

```math
error_D (h) \equiv \underset{x \in D}{\text{Pr}} [f(x) \neq h(x)]
```

The **sample error** of $h$ with respect to target function $f$ and data sample $S$ is the proportion of examples $h$ misclassifies

```math
error_S (h) \equiv \frac{1}{n} \sum_{x \in S} \delta(f(x) \neq h(x))
```

Where $\delta(f(x) \neq h(x))$ is 1 if $f(x) \neq h(x)$, and 0 otherwise.

How well does $error_{S}(h)$ estimate $error_{D}(h)$ ?

## Problems Estimating Error

1. $Bias$: If $S$ is training set, $error_{S}(h)$ is optimistically biased

```math
bias \equiv E[error_{S}(h)] - error_{D}(h)
```

  For unbiased estimate, $h$ and $S$ must be chosen independently

2. $Variance$: Even with unbiased $S$, $error_{S}(h)$ may still $vary$ from $error_{D}(h)$

## Example

Hypothesis $h$ misclassifies 12 of the 40 examples in $S$

```math
error_{S}(h) = \frac{12}{40} = .30
```

What is $error_{D}(h)$ ?

## Estimators

Experiment:

1. choose sample $S$ of size $n$ according to distribution $D$
2. measure $error_{S}(h)$

$error_{S}(h)$ is a random variable (i.e., result of an experiment)

$error_{S}(h)$ is an unbiased $estimator$ for $error_{D}(h)$

Given observed $error_{S}(h)$, what can we comclude about $error_{D}(h)$ ?

## Confidence Intervals

If

- $S$ contains $n$ examples, drawn independently of $h$ and each other
- $n \geq 30$

Then

- With approximately 95% probability, $error_{D}(h)$ lies in interval

```math
error_{S}(h) \pm 1.96 \sqrt{\frac{error_{S}(h)(1 - error_{S}(h))}{n}}
```

If

- $S$ contains $n$ examples, drawn independently of $h$ and each other
- $n \geq 30$

Then

- With approximately N% probability, $error_{D}(h)$ lies in interval

```math
error_{S}(h) \pm z_{N} \sqrt{\frac{error_{S}(h)(1 - error_{S}(h))}{n}}
```

  where

```math
$$
\begin{array}{|c|c|c|c|c|c|c|c|}
N\%: & 50\% & 68\% & 80\% & 90\% & 95\% & 98\% & 99\% \\
z_N: & 0.67 & 1.00 & 1.28 & 1.64 & 1.96 & 2.33 & 2.58
\end{array}
$$
```

## $error_{S}(h)$ is a Random Variable

Rerun the experiment with different randomly drawn $S$ (of size $n$)

Probability of observing $r$ misclassified examples:

<img width="502" alt="Screenshot 2568-04-26 at 13 19 46" src="https://github.com/user-attachments/assets/821bff81-446e-4ce4-a221-167b047bb19c" />

```math
P(r) = \frac{n!}{r!(n-r)!} error_{D}(h)^{r}(1 - error_{D}(h))^{n-r}
```

## Binomial Probability Distribution

<img width="502" alt="Screenshot 2568-04-26 at 13 19 46" src="https://github.com/user-attachments/assets/821bff81-446e-4ce4-a221-167b047bb19c" />

```math
P(r) = \frac{n!}{r!(n-r)!} p^{r} (1 - p)^{n-r}
```

Probability $P(r)$ of $r$ heads in $n$ coin flips, if $p$ = $\Pr(heads)$

- Expected, or mean value of $X$, $E[X]$, is

```math
E[X] \equiv \sum_{i = 0}^{n} iP(i) = np
```

- Variance of $X$ is

```math
Var(X) \equiv E[(X - E[X])^{2}] = np(1-p)
```

- Standard deviation of $X$, $\sigma_{X}$, is

```math
\sigma_{X} \equiv \sqrt{E[(X - E[X])^{2}]} = \sqrt{np(1 - p)}
```
