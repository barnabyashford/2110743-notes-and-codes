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
