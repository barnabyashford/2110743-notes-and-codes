# Learning from examples

For this week, the chapter includes:

- General-to-specific ordering over hypotheses
- Version space & candidate elimination algorithm
- Inductive[^1] bias

[^1]: Inductive, Induction &rarr; to induct: to extract a assumptive pattern by generalising from what one has learned.

## Data

**Data name**: $EnjoySport$

**Features and possible value**:

| Feature | Number of possible values | Possible values | Note |
| :---: | :---: | :---: | :---: |
| $Sky$ | 3 | $Sunny$, $Rainy$, $Cloudy$ |  |
| $Temp$ | 2 | $Warm$, $Cold$ |  |
| $Humid$ | 2 | $High$, $Normal$ |  |
| $Wind$ | 2 | $Strong$, $Weak$ |  |
| $Water$ | 2 | $Warm$, $Cool$ |  |
| $Forecst$ | 2 | $Same$, $Change$ |  |
| $EnjSpt$ | 2 | $Yes$: $+$, $No$: $-$ | Target Attribute<br>Binary Classification |

All example possibility: 

$$ 3 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 = 96\text{ possibilities} $$

**Examples**:

| $Sky$ | $Temp$ | $Humid$ | $Wind$ | $Water$ | $Forecst$ | $EnjSpt$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $Sunny$ | $Warm$ | $Normal$ | $Strong$ | $Warm$ | $Same$ | $+$ |
| $Sunny$ | $Warm$ | $High$ | $Strong$ | $Warm$ | $Same$ | $+$ |
| $Rainy$ | $Cold$ | $High$ | $Strong$ | $Warm$ | $Change$ | $-$ |
| $Sunny$ | $Warm$ | $High$ | $Strong$ | $Cool$ | $Change$ | $+$ |

> One row : one example

What we want to learn from these examples: what is the general concept?

## Representing Hypotheses

Hypothesis ($h$): conjunction ($\land$) of constraints on attributes (all attributes as a logical value (i.e., True/False) connected with logical `AND` operation.)

Each constraint can be:
- Specific value (e.g., $"Water = Warm"$)
- Don't care (e.g., $?$)
  - the value can be anything
- no value allowed (e.g., $"Water = \emptyset"$)
  - Cannot be any value, if the feature with this constraint even if other features align with the hypotheis, the example will automatically be of "negative" ($-$) class.

For example:

| $Sky$ | $Temp$ | $Humid$ | $Wind$ | $Water$ | $Forecst$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $Sunny$ | $?$ | $?$ | $Strong$ | $?$ | $Same$ |

## Prototypical concept learning task

"**Concept Learning**"

**Given**:
- Instances $X$: Possible days, descirbed by features: $Sky$, $Temp$, $Humid$, $Wind$, $Water$, and $Forecst$
  - **Note**: "Instance" does not include target value, but it basically means a datapoint.
- Target Function $c$: $EnjoySport : X \rightarrow \\{0,1\\}$
- Hypothese $H$: Conjunctions ($\land$) of literals (constraint)
- Training examples $D$: Positive and negative examples of the target function

$$ \langle x_1, c(x_1),...(x_m, c(x_m)\rangle $$

**Determine (learn)**: A hypothesis $h$ in $H$ such that $h(x) = c(x)$ for all $x$ in $D$.

## The inductive[^1] learning hypothesis

"Any hypothesis found to approximate the target function well over a sufficiently large set of training examples will also approximate the target function well over other unobserved examples"

A hypothesis that if one is to train a model they must have in mind that, for your model to be deem as a "good" model, it must perform well not only with its training and/or test data, but on other unseen data as well.

## Instance, Hypotheses, and More general than

To visualise what Instance and Hypothesis "space" means:

<img width="592" alt="Screenshot 2568-03-01 at 22 56 37" src="https://github.com/user-attachments/assets/a206ca13-31f6-410b-98ad-256b219c58a0" />

> in the image $Light$ basically means $Weak$.

From the spaces above, now we can see how instance space relates to hypothesis space.

**recap**:

All example possibility: 

$$ 3 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 = 96\text{ possibilities} $$

We can see that, if we were to account the possibility of a feature as a constraint, each feature will now have two other possble values: $?$ and $\emptyset$

Therefore, the possibility of all hypothesis is: 

$$ 5 \times 4 \times 4 \times 4 \times 4 \times 4 \times 4 = 5120\text{ possibilities} $$

However, if we consider the purpose of the "no value allowed" we will see that, no matter which feature has its constraint as $\emptyset$, or how many features in a hypothesis has that constraint, it will mean the very same. Therefore, if we account the constraint as just one possibility for its meaning. We will now have:

- Syntactic hypothesis possibilities:

$$ 5 \times 4 \times 4 \times 4 \times 4 \times 4 \times 4 = 5120\text{ possibilities} $$

- Semantic hypothesis possibilities (The added $+1$ is the "no value allowed" constraint)

$$ (4 \times 3 \times 3 \times 3 \times 3 \times 3 \times 3) + 1 = 973\text{ possibilities} $$

> Now if we consider real-world implementation that, a feature can have up to tens of possible values, and hundreds of features. The possibilty of examples will be very large, causing the possibility of all hypothesis will be even larger.

In hypothesis space, spatial meaning from top to buttom range from: most specific to most general.

<img width="309" alt="Screenshot 2568-03-01 at 23 18 12" src="https://github.com/user-attachments/assets/e462ece5-401d-43bb-9458-7809767023b9" />

From the image above, we can see that $h_2$ only specifies one feature, while $h_1$ and $h_3$. We can say that $h_2$ is more general than $h_1$ and $h_3$.

## $\text{Find-S}$ Algorithm

**Steps**:

1. Initialise $h$ to the most specific hypothesis in $H$.
2. For each positive training instance $x$
  - For each attribute constraint $a_i$ in $h$  
    If the constraint $a_i$ in $h$ is satisfied by $x$  
    Then do nothing  
    Else replace $a_i$ in $h$ by the next more general constraint that is satisfied by $x$
3. Output hypothesis $h$

basically, what it does is just change from $\emptyset$ to any specific value, and possibly to $?$. We can, however, change from that to do more to fit our own use case in real-world implmentation.

From our example data:

<img width="566" alt="Screenshot 2568-03-01 at 23 30 56" src="https://github.com/user-attachments/assets/96ec0963-5585-49ef-918c-4e6fd375e897" />

We can see that the image follows the $\text{Find-S}$ algorithm by initialising the most specific $h_0$ then iterate over the data on the left, adjust the hypothesis to fit any positive data it finds.

### Limitations of $\text{Find-S}$ Algorithm

- Can't tell whether it has finished learning. We don't really know if we input more data, will the algorithm try to generalise the hypothesis or will it do nothing.
- Can't tell if the data is inconsistent. If the data contains noise from error in recording, the algorithm won't say a thing, and continue to modify the hypothesis.
- Pick the most specific $h$. Even if there are more general hypothesis, which could possibly work better, the algorithm would not choose it if there is a more specific hypothesis.
- There might be more that one maximally specific $h$, depending on $H$.

## Version Spaces

<img width="543" alt="Screenshot 2568-03-02 at 01 10 03" src="https://github.com/user-attachments/assets/7da625b3-d9a1-496d-b8cf-86a715effb03" />

Idea: A hypothesis $h$ is **consistent** (accepts only positive examples) with a set of training examples $D$ of target concept $c$ if and only if $h(c) = c(x)$ for each training example $\langle x, c(x)\rangle$ in $D$.

$$ Consistent(h,D) \equiv (\forall \langle x, c(c)\rangle \in D) h(x) = c(x) $$

Actually a subset of hypothesis space. The **version space** $VS_{H,D}$ with respect to hypothesis space $H$ and training examples $D$, isthe subset of hypotheses from $H$ consistent with all training examples in $D$.

$$ VS_{H,D} \equiv \\{h \in H \mid Consistent(h,D)\\} $$

### The $\text{List-Then-Eliminate}$ Algorithm

1. $VersionSpace$ &larr; a list containing every hypothesis in $H$
2. For each training example (no matter even if the example is negative or positive), $\langle x, c(x)\rangle$  
  remove from $VersionSpace$ any hypothesis $h$ for which $h(x) \neq c(x)$
3. Output the list of hypotheses in $VersionSpace$

**Problem**: This is an exhaustive algorithm. It requires us to list all possible hypothesis. Which is impossible to be done in real-world implementation where there are thousands of possible example and expontially larger possibilities of hypotheses.

**How do we fix that?**: We exploit the ordinal nature of version space. That way, we can find a way to represent the hypothesis space without actually list all possibility.

### Representing Version Spaces

The **General boundary**, $G$, of version space $VS_{H,D}$ is the set of its maximally general members

The **Specific boundary**, $S$, of version space $VS_{H,D}$ is the set of its maximally specific members

Every member of the version space lies between these boundaries

$$ VS_{H,D} = \\{ h \in H \mid (\exists s \in S) (\exists g \in G) (g \ge h \ge s)\\} $$

where $x \ge y$ means $x$ is more general or equal to $y$

### Candidate Elimination Algorithm

$G$ &larr; maximally general hypotheses in $H$: $\\{\langle ?,?,?,?,?,? \rangle\\}$

$S$ &larr; maximally specific hypotheses in $H$: $\\{\langle \emptyset,\emptyset,\emptyset,\emptyset,\emptyset,\emptyset \rangle\\}$

For each training example $d$, do

If $d$ is a positive example (remove inconsistent $g$, generalise $S$)

- Remove from $G$ any hypothesis inconsistent with $d$
- For each constraint $a_i$ in $s$ in $S$ that is not consistent with $d$
  - change $a_i$ to $?$

If $d$ is a negative example (remove inconsistent $s$, specialise $G$)

- Remove from $S$ any hypothesis inconsistent with $d$
- For each hypothesis $g$ in $G$ that is not consistent with $d$
  - Remove $g$ from $G$
  - Add to $G$ all minimal specialisations $h$ of $g$ such that
    1. $h$ is consistent with $d$, and
    2. some member of $S$ is more specific than $h$
  - Remove from $G$ any hypothesis that is less general than another hypothesis in $G$

#### Example Tracing

##### Data

| $Sky$ | $Temp$ | $Humid$ | $Wind$ | $Water$ | $Forecst$ | $EnjSpt$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $Sunny$ | $Warm$ | $Normal$ | $Strong$ | $Warm$ | $Same$ | $+$ |
| $Sunny$ | $Warm$ | $High$ | $Strong$ | $Warm$ | $Same$ | $+$ |
| $Rainy$ | $Cold$ | $High$ | $Strong$ | $Warm$ | $Change$ | $-$ |
| $Sunny$ | $Warm$ | $High$ | $Strong$ | $Cool$ | $Change$ | $+$ |

##### Step 0: initialisation of $G$ and $S$

space:

$$ S_0 = $\\{\langle \emptyset,\emptyset,\emptyset,\emptyset,\emptyset,\emptyset \rangle\\} $$

$$ G_0 = \\{\langle ?,?,?,?,?,? \rangle\\} $$

##### Step 1

pointed example:

- $x_1 = \langle Sunny, Warm, Normal, Strong, Warm, Same \rangle$
- $c(x_1) = +$ &rarr; generalise $S$

space:

$$ S_0 = \\{\langle \emptyset,\emptyset,\emptyset,\emptyset,\emptyset,\emptyset \rangle\\} $$

$$ \downarrow $$

$$ S_1 = $\\{\langle Sunny, Warm, Normal, Strong, Warm, Same \rangle\\} $$

$$ G_1 = G_0 = \\{\langle ?,?,?,?,?,? \rangle\\} $$

> $G_0$ is already consistent with $x_1$

##### Step 2

pointed example:

- $x_2 = \langle Sunny, Warm, High, Strong, Warm, Same \rangle$
- $c(x_2) = +$ &rarr; generalise $S$

space:

$$ S_0 = \\{\langle \emptyset,\emptyset,\emptyset,\emptyset,\emptyset,\emptyset \rangle\\} $$

$$ \downarrow $$

$$ S_1 = \\{\langle Sunny, Warm, Normal, Strong, Warm, Same \rangle\\} $$

$$ \downarrow $$

$$ S_2 = \\{\langle Sunny, Warm, ?, Strong, Warm, Same \rangle\\} $$

$$ G_2 = G_1 = G_0 = \\{\langle ?,?,?,?,?,? \rangle\\} $$

##### Step 3

pointed example:

- $x_3 = \langle Rainy, Cold, High, Strong, Warm, Change \rangle$
- $c(x_3) = -$ &rarr; specialise $G$

space:

$$ S_0 = \\{\langle \emptyset,\emptyset,\emptyset,\emptyset,\emptyset,\emptyset \rangle\\} $$

$$ \downarrow $$

$$ S_1 = $\\{\langle Sunny, Warm, Normal, Strong, Warm, Same \rangle\\} $$

$$ \downarrow $$

$$ S_3 = S_2 = \\{\langle Sunny, Warm, ?, Strong, Warm, Same \rangle\\} $$

$$ G_3 = \\{\langle Sunny,?,?,?,?,? \rangle, \\{\langle ?, Warm,?,?,?,? \rangle, \langle ?,?,?,?,?,Same \rangle\\} $$

$$ \uparrow $$

$$ G_2 = G_1 = G_0 = \\{\langle ?,?,?,?,?,? \rangle\\} $$

> $S_2$ is already consistent with $x_3$ because it denies $x_3$, and $c(x_3) = -$

> To specialise $G$, in adding $h$, we have to add a minimal specialisation in each feature "in accordance to the current $S$". According to this, there is also supposed to be $\\{\langle ?,?,?, Strong,?,? \rangle, \langle ?,?,?,?,Warm,? \rangle\\}$ as well, but they are removed as they are not consistent with $x_3$ (Both accept $x_3$ while $x_3$ is a negative example.)

##### Step 4

pointed example:

- $x_4 = \langle Sunny, Warm, High, Strong, Cool, Change \rangle$
- $c(x_4) = +$ &rarr; generalise $S$

space:

$$ S_0 = \\{\langle \emptyset,\emptyset,\emptyset,\emptyset,\emptyset,\emptyset \rangle\\} $$

$$ \downarrow $$

$$ S_1 = \\{\langle Sunny, Warm, Normal, Strong, Warm, Same \rangle\\} $$

$$ \downarrow $$

$$ S_3 = S_2 = \\{\langle Sunny, Warm, ?, Strong, Warm, Same \rangle\\} $$

$$ \downarrow $$

$$ S_4 = \\{\langle Sunny, Warm, ?, Strong, ?, ? \rangle\\} $$

$$ G_4 = \\{\langle Sunny,?,?,?,?,? \rangle, \\{\langle ?, Warm,?,?,?,? \rangle\\} $$

$$ \uparrow $$

$$ G_3 = \\{\langle Sunny,?,?,?,?,? \rangle, \\{\langle ?, Warm,?,?,?,? \rangle, \langle ?,?,?,?,?,Same \rangle\\} $$

$$ \uparrow $$

$$ G_2 = G_1 = G_0 = \\{\langle ?,?,?,?,?,? \rangle\\} $$

##### Result

To summarise the current $S$ and $G$, we have to find a set of $h$ such that they illustrate the journey from $G$ to $S$. That is,:

For each $g$ in the latest $G$, do

- for each $a_i$ of $g$ that is more general than $a_i$ of $s$ in $S$
  - change $a_i$ in $g$ to match $a_i$ of $s$ in $S$

For now we have:

$$ S_4 = \\{\langle Sunny, Warm, ?, Strong, ?, ? \rangle\\} $$

$$ G_4 = \\{\langle Sunny,?,?,?,?,? \rangle, \\{\langle ?, Warm,?,?,?,? \rangle\\} $$

in order to follow the finalising algorithm here are the steps:

1. $\langle Sunny,?,?,?,?,? \rangle$
  - change $a_{Temp}$ from $?$ to $Warm$: $\langle Sunny,Warm,?,?,?,? \rangle$
  - change $a_{Wind}$ from $?$ to $Strong$: $\langle Sunny,?,?,Strong,?,? \rangle$
2. $\langle ?, Warm,?,?,?,? \rangle$
  - change $a_{Sky}$ from $?$ to $Sunny$: $\langle Sunny,Warm,?,?,?,? \rangle$
    - **duplicate**
  - change $a_{Wind}$ from $?$ to $Strong$: $\langle ?,Warm,?,Strong,?,? \rangle$

here is the finalised spaced from what we have

<img width="543" alt="Screenshot 2568-03-02 at 01 10 03" src="https://github.com/user-attachments/assets/7da625b3-d9a1-496d-b8cf-86a715effb03" />

**Note**: As there are not enough examples in the data to finalise into one $h$, the current version space is not converged.

### What Next Training Example?

From the finalised space we have got:

<img width="543" alt="Screenshot 2568-03-02 at 01 10 03" src="https://github.com/user-attachments/assets/7da625b3-d9a1-496d-b8cf-86a715effb03" />

We can use the remaining three hypothesis to "ask" the model, what kind of example do they need to take the next step. Here is how we can do it:

1. Look at the space, see the correlation between the elements and choose one if there are many. Here, there are:
  - $a_{Wind} = Strong$ in $S$, $\langle Sunny,?,?,Strong,?,? \rangle$, and $\langle ?,Warm,?,Strong,?,? \rangle$
  - $a_{Wind} = ?$ in $G$ and $\langle Sunny,Warm,?,?,?,? \rangle$ 

2. Try inputting a new example, say, $\langle Sunny, Warm, Normal, Weak, Warm, Same\rangle$ along with its label (by a labeler). With this we can remove many remaining hypotheses that are not consistent with the new example.

### We can use the unconverged $h$ as well!

**Given**:
- $\langle Sunny, Warm, Normal, Strong, Cool, Change\rangle$
- $\langle Rainy, Cool, Normal, Weak, Warm, Same\rangle$
- $\langle Sunny, Warm, Normal, Weak, Warm, Same\rangle$

**Use every remaining $h$ to classify**
- If $S$ and $G$ agrees, there is no need to see the finalsed hypotheses in the between space.
- If $S$ and $G$ disagrees, try the $h$ in between, and output the classification result that most $h$ (including the ones in $S$ and $G$) say. Moreover, if the classification result came at a stalemate, output "unknown".

**Output**:
- $\langle Sunny, Warm, Normal, Strong, Cool, Change\rangle$ &rarr; $+$
- $\langle Rainy, Cool, Normal, Weak, Warm, Same\rangle$ &rarr; $-$
- $\langle Sunny, Warm, Normal, Weak, Warm, Same\rangle$ &rarr; $unknown$

## What Justifies this Inductive[^1] Leap

**Given**:

$$ + \langle Sunny, Warm, Normal, Strong, Cool, Change \rangle $$

$$ + \langle Sunny, Warm, Normal, Weak, Warm, Same \rangle $$

**Resulting Hypotheses**:

$$ S : \langle Sunny, Warm, Normal, ?, ?, ? \rangle $$

Why believe we can classify the unseen?

$$ \langle Sunny, Warm, Normal, Strong, Warm, Same \rangle $$

This "induction leap" is caused by a "bias". The decision on the unseen example to be as what you have generalised into your very own bias, learned from what you have seen.

This is like when you were young, you thought what makes an animal a "Bird" is the fact that they can fly, but not all bird can fly! That is your bias! While not true, it can still help you decide correctly in some cases.

## An UNBiased Learner

From our data, we have calculated all possible example. It was 96.

**recap**:

All example possibility: 

$$ 3 \times 2 \times 2 \times 2 \times 2 \times 2 \times 2 = 96\text{ possibilities} $$

For each possbile pattern of example, we can either assign it as $+$ or $-$. That leave us with all possible classifier of:

$$ 2^{96} $$

An unknown amount of millions. We cannot possible create all of them! Therefore, with the hypothesis space we have created with our defined features there are, in total, 5210 possibilities.

**recap**:

- Syntactic hypothesis possibilities:

$$ 5 \times 4 \times 4 \times 4 \times 4 \times 4 \times 4 = 5120\text{ possibilities} $$

We can see that there are limits to the actual possibilities, causing us to maximally find 5210 ways of representation from 2<sup>96</sup>. That is caused by our own bias in determining the learning algorithm that allows us to only find 5120 classifier from the whoping amount of 2<sup>96</sup>.

Before we only allow, conjuntion ($\land$). This is our bias. To move away from our bias, we have to try other operation as well.

Idea: Choose $H$ that expresses every teachable concept (i.e., $H$ is the power set of $X$)

Consider $H^\prime$ = disjunctions, conjunctions,
negations over previous $H$. e.g.,

$$ \langle Sunny, Warm, Normal, ?, ?, ? \rangle \lor \neg \langle ?, ?, ?, ?, ?, Change\rangle $$

What are $S$, $G$ in this case?

- say

$$ X_+ = \\{ x_1, x_2, x_3 \\} $$

$$ X_- = \\{ x_4, x_5 \\} $$

- The UNBiased hypotheses

$$ S \leftarrow \\{ x_1 \lor x_2 \lor x_3 \\} $$
$$ G \leftarrow \\{ x_4 \lor x_5 \\} $$

We can see that the learner from this idea leaned from merely "memorising" all training input, this kind of learning is called **Rote Learning**.

The fact that this kind of learning can learn anything as it is unbiased does not entail its invincibility. Being completely unbias means that it is unable to make inductive[^1] leap, making very vulnerable to unseen data.

## Inductive[^1] Bias

Consider

- concept learning algorithm $L$
- instances $X$, target concept $c$
- training examples $D_c = \\{ \langle x, c(x)\rangle\\}$
- let $L(x_i, D_c)$ denote the classification assigned to the instance $x_i$ by $L$ after training on data $D_c$.

**Definition**:

The inductive[^1] bias of $L$ is any minimal set of assertions $B$ such that for any target concept $c$ and corresponding training examples $D_c$

$$ (\forall x_i \in X)[( B \land D_c \land x_i) \vdash L(x_i,D_c) $$

where $A \vdash B$ means $A$ logically entails $B$

## Inductive Systems and Equivalent Deductive[^2] Systems

<img width="613" alt="Screenshot 2568-03-02 at 03 39 34" src="https://github.com/user-attachments/assets/c60274cc-4cfd-4a28-932f-186c01fb8cfe" />

[^2]: Deductive systems works with fact, while inductive systems works with pattern and trends of the training data. The "Equivalent Deductive System" is meant to imitate the Inductive System to prove its bias by taking the same input with additional bias value.

## Three Learner with Different Biases

- $\text{Rote learner}$: Store examples, Classify x if and only if it matches previously observed example.
  - No Bias, it just classifies as what it remembers
- $\text{Version space candidate elimination algorithm}$
  - Bias: learnable only if $c \in H$
- $\text{Find-S}$
  - Bias: learnable only if$c \in H$ and prefer only specific $h$.

## Summary Points

1. Concept learning as search through $H$
2. General-to-specific ordering over $H$
3. Version space candidate elimination algorithm
4. $S$ and $G$ boundaries characterize learner's uncertainty
5. Learner can generate useful queries
6. Inductive leaps possible only if learner is biased
7. Inductive learners can be modelled by equivalent deductive systems
