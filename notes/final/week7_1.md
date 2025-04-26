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

## Normal Distribution Approximates Binomial

$error_{S}(h)$ follows a *Binomial* distribution, with

- mean $\mu_{error_{S}(h)} = error_{D}(h)$
- standard deviation $\sigma_{error_{S}(h)}$

```math
\sigma_{error_{S}(h)} = \sqrt{\frac{error_{D}(h)(1 - error_{D}(h))}{n}}
```

Approximate this by a *Normal* distribution with

- mean $\mu_{error_{S}(h)} = error_{D}(h)$
- standard deviation $\sigma_{error_{S}(h)}$

```math
\sigma_{error_{S}(h)} \approx \sqrt{\frac{error_{D}(h)(1 - error_{D}(h))}{n}}
```

## Normal Probability Distribution

<img width="385" alt="Screenshot 2568-04-26 at 15 33 09" src="https://github.com/user-attachments/assets/eeb2b69f-8d5d-4fcd-bd77-745c3924db8b" />

```math
p(x) = \frac{1}{\sqrt{2\pi\sigma^{2}}} e^{-\frac{1}{2} \left( \frac{x - \mu}{\sigma} \right)^{2}}
```

The probability that $X$ will fall into the interval $(a,b)$ is given by

```math
\int_{a}^{b} p(x)dx
```

- Expected, or mean value of $X$, $E[X]$, is

```math
E[X] = \mu
```

- Variance of $X$ is

```math
Var(X) = \sigma^{2}
```

- Standard distribution of $X$, $\sigma_{X}$, is

```math
\sigma_{X} = \sigma
```

## Normal Probability Distribution

<img width="378" alt="Screenshot 2568-04-26 at 15 39 50" src="https://github.com/user-attachments/assets/06effc5f-a4e2-4b53-ab82-19e20789c3b9" />

80% of area (probability) lies in $\mu \pm 1.28\sigma$

N% of area (probability) lies in $\mu \pm z_{N}\sigma$

```math
$$
\begin{array}{|c|c|c|c|c|c|c|c|}
N\%: & 50\% & 68\% & 80\% & 90\% & 95\% & 98\% & 99\% \\
z_N: & 0.67 & 1.00 & 1.28 & 1.64 & 1.96 & 2.33 & 2.58
\end{array}
$$
```

## Confidence Intervals, More Correctly

If

- $S$ contains $n$ examples, drawn independently of $h$ and each other
- $n \geq 30$

Then

- With approximately 95% probability, $error_{S}(h)$ lies in interval

```math
error_{D}(h) \pm 1.96 \sqrt{\frac{error_{D}(h)(1 - error_{D}(h))}{n}}
```

equivalently, error_{D}(h) lies in interval

```math
error_{S}(h) \pm 1.96 \sqrt{\frac{error_{D}(h)(1 - error_{D}(h))}{n}}
```

which is approximately

```math
error_{S}(h) \pm 1.96 \sqrt{\frac{error_{S}(h)(1 - error_{S}(h))}{n}}
```

## Central Limit Theorem

Consider a set of independent, identically distributed random variables $Y_{1} \dots Y_{n}$, all governed by an arbitrary probability distribution with mean $\mu$ and finite variance $\sigma^{2}$. Define the sample mean,

```math
\bar{Y} \equiv \frac{1}{n} \sum_{i = 1}^{n} Y_{i}
```

**Central Limit Theorem.** As $n \rightarrow \infty$, the distribution governing $\bar{Y}$ approaches a Normal distribution, with mean $\mu$ and variance \frac{\sigma^{2}}{n} .

## Calculating Confidence Intervals

1. Pick parameter $p$ to estimate
  - $error_{D}(h)$
2. Choose an estimator
  - $error_{S}(h)$
3. Determine probability distribution that governs estimator
  - $error_{S}(h)$ governed by Binomial distribution, approximated by Normal when $n \geq 30$
4. Find interval $(L,U)$ such that N% of probability mass falls in the interval
  - USe table of $z_{N}$ values

## Difference Between Hypotheses

Test $h_{1}$ on sample $S_{1}$, test $h_{2}$ on $S_{2}$

1. Pick parameter to estimate

```math
d \equiv error_{D}(h_{1}) - error_{D}(h_{2})
```

2. Choose an estimator

```math
\hat{d} \equiv error_{S_{1}}(h_{1}) - error_{S_{2}}(h_{2})
```

3. Determine probability distribution that governs estimator

```math
\sigma_{\hat{d}} \approx \sqrt{\frac{error_{S_{1}}(h_{1})(1 - error_{S_{1}}(h_{1}))}{n_{1}} + \frac{error_{S_{2}}(h_{2})(1 - error_{S_{2}}(h_{2}))}{n_{2}}}
```

4. Find interval $(L,U)$ such that N% of probability mass falls in the interval

```math
\hat{d} \pm z_{N} \sqrt{\frac{error_{S_{1}}(h_{1})(1 - error_{S_{1}}(h_{1}))}{n_{1}} + \frac{error_{S_{2}}(h_{2})(1 - error_{S_{2}}(h_{2}))}{n_{2}}}
```

## Paired $t$ test to compare $h_{A},h_{B}$

1. Partition data into $k$ disjoint test sets $T_{1},T_{2},\dots,T_{k}$ of equal size, where this size is at least 30.
2. For $i$ from 1 to $k$, do

```math
\delta_{i} \leftarrow error_{T_{i}}(h_{A}) - error_{T_{i}}(h_{B})
```

3. Return the value $\bar{\gamma}$, where

```math
\bar{\delta} \equiv \frac{1}{k} \sum_{i = 1}^{k} \delta_{i}
```

N% confidence interval estimate for $d$:

```math
\bar{\delta} \pm t_{N,k-1} s_{\bar{\delta}}
```

```math
s_{\bar{\delta}} \equiv \sqrt{\frac{1}{k(k - 1)} \sum_{i = 1}^{k} (\delta_{i} - \bar{\delta})^{2}}
```

Note $\delta_{i}$ approximately Normally distributed

## Comparing learning algorithms $L_{A}$ and $L_{B}$

What we'd like to estimate:

```math
E_{S \subset D} [ error_{D} (L_{A}(S)) - error_{D}(L_{B}(S))]
```

where $L(S)$ is the hypothesis output by learner $L$ using training set $S$

i.e., the expected difference in true error between hypotheses output by learners $L_{A}$ and $L_{B}$, when trained using randomly selected training sets $S$ drawn according to distribution $D$.

But, given limited data $D_{0}$, what is a good estimator?

- could partition $D_{0}$ into training set $S_{0}$ and test set $T_{0}$, and measure

```math
error_{T_{0}} (L_{A}(S_{0})) - error_{T_{0}} (L_{B}(S_{0}))
```

- even better, repeat this many times and average the results (next slide)

1. Partition data $D_{0}$ into $k$ disjoint test sets $T_{1}, T_{2}, \dots, T_{k}$ of equal size, where this size is at least 30.

2. For $i$ from 1 to $k$, do  
use $T_{i}$ for the test set, and the remaining data for training set $S_{i}$

  - $S_{i} \leftarrow \\{ D_{0} - T_{i} \\}$
  - $h_{A} \leftarrow L_{A}(S_{i})$
  - $h_{B} \leftarrow L_{B}(S_{i})$
  - $\delta_{i} \leftarrow error_{T_{i}}(h_{A}) - error_{T_{i}}(h_{B})$

3. Return the value $\bar{\delta}$, where

```math
\bar{\delta} \equiv \frac{1}{k} \sum_{i = 1}^{k} \delta_{i}
```

Notice we'd like to use the paired $t$ test on $\bar{\delta}$ to obtain a confidence interval

but not really correct, because the training sets in this algorithm are not independent (they overlap!)

more correct to view algorithm as producing an estimate of

```math
E_{S \subset D_{0}} [ error_{D} (L_{A}(S)) - error_{D} L_{B}(S))]
```

instead of

```math
E_{S \subset D} [ error_{D} (L_{A}(S)) - error_{D} L_{B}(S))]
```

but even this approximation is better than no comparison.
