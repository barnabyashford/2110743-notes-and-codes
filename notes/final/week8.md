# Bayesian Learning (2)

## Relation to Concept Learning

Consider our usual concept learning task

- instance space $X$, hypothesis space $H$, training examples $D$
- consider the $FindS$ learning algorithm (outputs most specific hypothesis from the version space $VS_{H,D}$)

What would Bayes rule produce as the $MAP$ hypothesis?

Does $FindS$ output a $MAP$ hypothesis?

Assume fixed set of instances $\langle x_1 , \dots , x_m \rangle$  
Assume $D$ is the set of classifications  $\langle c(x_1) , \dots , c(x_m) \rangle$

Choose $P(D \mid h)$ *noise-free:

- $P(D \mid h) = 1$ if $h$ consistent with $D$
- $P(D \mid h) = 0$ otherwise

Choose $P(h)$ to be *uniform* distribution

- $P(h) = \frac{1}{|H|}$ for all $h$ in $H$

Then,

```math
P(h \mid D) = \begin{cases} \frac{1}{|VS_{H,D}|} & \text{if } h \text{ is consistent with } D \\ 0 & \text{otherwise} \end{cases}
```
> from $P(h \mid D) =\arg \max_{h \in H} \frac{P(D \mid h)P(h)}{P(D)} = \arg \max_{h \in H} \frac{1 \cdot \frac{1}{|H|}}{\frac{|VS_{H,D}|}{|H|}} = \arg \max_{h \in H} \frac{1}{|VS_{H,D}|}$

<img width="537" alt="Screenshot 2568-04-26 at 20 51 00" src="https://github.com/user-attachments/assets/f4a971ba-5c09-4235-8605-4ee07f575fb5" />

> Evolution of Posterior Probabilities
> At first, all hypothesis seems to have the same probability.  
> If we put them to test (on a dataset), the hypotheses that does not fit the data have their probability dropped to 0, while the ones that fits have theirs increased.  
> The same happens if we adds another dataset.

## Characterizing Learning Algorithms by Equivalent MAP Learners

<img width="491" alt="Screenshot 2568-04-26 at 20 51 59" src="https://github.com/user-attachments/assets/c45e3920-e088-4925-8f9b-e19e3f1d1017" />
