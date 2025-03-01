# Decision Tree

This session talks about:
- Decision tree representation
- ID3 learning algorithm
- Entropy, Information gain
- Overfitting

## Data

**Data name**: $PlayTennis$

**Features and possible value**:

| Feature | Number of possible values | Possible values | Note |
| :---: | :---: | :---: | :---: |
| $Outlook$ | 3 | $Sunny$, $Rain$, $Overcast$ |  |
| $Temperature$ | 3 | $Hot$, $Mild$, $Cool$ |  |
| $Humidity$ | 2 | $High$, $Normal$ |  |
| $Wind$ | 2 | $Strong$, $Weak$ |  |
| $PlayTennis$ | 2 | $Yes$: $+$, $No$: $-$ | Target Attribute<br>Binary Classification |

> Here we exclude: $Day$ as it is just an identifying attribute, a unique key to each datapoint.

All example possibility: 

$$ 3 \times 3 \times 2 \times 2 = 36\text{ possibilities} $$

**Examples**:

| $Day$ | $Outlook$ | $Temperature$ | $Humidity$ | $Wind$ | $PlayTennis$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $D1$ | $Sunny$ | $Hot$ | $High$ | $Weak$ | $-$ |
| $D2$ | $Sunny$ | $Hot$ | $High$ | $Strong$ | $-$ |
| $D3$ | $Overcast$ | $Hot$ | $High$ | $Weak$ | $+$ |
| $D4$ | $Rain$ | $Mild$ | $High$ | $Weak$ | $+$ |
| $D5$ | $Rain$ | $Cool$ | $Normal$ | $Weak$ | $+$ |
| $D6$ | $Rain$ | $Cool$ | $Normal$ | $Strong$ | $-$ |
| $D7$ | $Overcast$ | $Cool$ | $Normal$ | $Strong$ | $+$ |
| $D8$ | $Sunny$ | $Mild$ | $High$ | $Weak$ | $-$ |
| $D9$ | $Sunny$ | $Cool$ | $Normal$ | $Weak$ | $+$ |
| $D10$ | $Rain$ | $Mild$ | $Normal$ | $Weak$ | $+$ |
| $D11$ | $Sunny$ | $Mild$ | $Normal$ | $Strong$ | $+$ |
| $D12$ | $Overcast$ | $Mild$ | $High$ | $Strong$ | $+$ |
| $D13$ | $Overcast$ | $Hot$ | $Normal$ | $Weak$ | $+$ |
| $D14$ | $Rain$ | $Mild$ | $High$ | $Strong$ | $-$ |

## Decision Tree for $PlayTennis$

<img width="517" alt="Screenshot 2568-03-02 at 05 49 42" src="https://github.com/user-attachments/assets/7827854e-9372-4830-b7cb-eb5256c7bf27" />

## A Tree to predict C-Section risk

Learned from medical records of 1000 women. Negative examples are C-sections.

```plaintext
[833+,167-] .83+ .17-
Fetal_Presentation = 1: [822+,116-] .88+ .12-
| Previous_Csection = 0: [767+,81-] .90+ .10-
| | Primiparous = 0: [399+,13-] .97+ .03-
| | Primiparous = 1: [368+,68-] .84+ .16-
| | | Fetal_Distress = 0: [334+,47-] .88+ .12-
| | | | Birth_Weight ‹ 3349: [201+,10.6-] .95+ .05-
| | | | Birth_Weight >= 3349: [133+,36.4-] .78+ .22-
| | | Fetal_Distress = 1: [34+,21-] .62+ .38-
| Previous_Csection = 1: [55+, 35-] .61+ .39-
Fetal_Presentation = 2: [3+,29-] .11+ .89-
Fetal_Presentation = 3: [8+,22-] .27+ .73-
```

## Decision Trees

Decision tree representation:

- Each internal node tests an attribute
- Each branch corresponds to attribute value
- Each leaf node assigns a classification

How would we represent:

- $\land$, $\lor$, $XOR$
- $(A \land B) \lor (C \land \neg D \land E)
- $M$ of $N$

### When to Consider Decision Trees

- Instances describable by attribute-value pairs
- Target function is discrete valued
- Disjunctive hypothesis may be required
- Possibly noisy training data

Examples:

- Equipment or medical diagnosis
- Credit risk analysis
- Modeling calendar scheduling preferences

### Top-Down Induction of Decision Trees

**Main loop**:

1. $A$ &larr; the "best" decision attribute for next $node$
2. Assign $A$ as decision attribute for $node$
3. For each value of $A$, create new descendant of $node$
4. Sort training examples to leaf nodes
5. If training examples perfectly classified, Then STOP, Else iterate over new leaf nodes

**Which attribute is best?**

<img width="501" alt="Screenshot 2568-03-02 at 06 02 31" src="https://github.com/user-attachments/assets/336258dd-eb20-4c9a-8fb6-dd4bb224354c" />

Probably `A1`, as the descendant nodes are more biased to either positive or negative examples, hinting the attribute's ability to separate data in a near-seamless manner.

### Entropy

<img width="307" alt="Screenshot 2568-03-02 at 06 04 52" src="https://github.com/user-attachments/assets/9e5587b9-905a-438f-8b5e-2b9b37db36fe" />

- $S$ is a sample of training examples
- $p_\oplus$ is the proportion of positive examples in $S$
- $p_\ominus$ is the proportion of negative examples in $S$
- Entropy measures the impurity of S

$$ Entropy(S) \equiv -p_\oplus log_2 p_\oplus -p_\ominus log p_\ominus $$

From the distribution graph, we can see that $Entropy(\cdot)$ can be as low as 0 and as high as 1. The way it increase or decrease is up to the proportion of the data, where the more biased the data is, the lower it gets, the less bias the data is, the higher it gets. If the data contains only members of a single class, $Entropy(\cdot)$ drops to zero, while if classes are balanced, $Entropy(\cdot)$ go right up to one.
