# Rule Learning

- Sequenctial Covering Algorithm
- FOIL
- Induction --
- 

## Learning Disjunctive Sets of Rules

Disjunctive rules: rules connected with $\land$.

Method 1: Learn decision tree, convert to rules

Method 2: Sequential (Rule-by-rule, not simultaneous) Covering (covers positive examples) Algorithm:

1. *Learn one rule* with high accuracy, any coverage (does not have to cover all positive examples, as another rule will be learned to cover more)
2. Remove positive examples covered by this rule
3. Repeat

## Sequential Covering Algorithm

$\text{SEQUENTIAL-COVERING} (TargetAttribute, Attributes, Examples, Threshold)$

- $Learned\\_Rules \leftarrow \\{\\}$
- $Rule \leftarrow \text{LEARN-ONE-RULE} (TargetAttribute, Attributes, Examples)$
- $while \  \{PERFORMANCE} (Rule, Examples) > Threshold \  do$
  - $Learned\\_Rules \leftarrow Learned\_Rules + Rule$
  - $Examples \leftarrow Examples - \\{ \text{examples correctly classified by } Rule \\}$
  - $Rule \leftarrow \text{LEARN-ONE-RULE} (TargetAttribute, Attributes, Examples)$
- $Learned\\_Rules \leftarrow sort \  Learned\\_Rules \  accord \  to \  \text{PERFORMANCE} \  Examples$
- $return \  Learned\\_Rules$

## Learn-One-Rule

1. Start from *most general rule* (a rule without any condition &rarr; accepts all)
2. Specialise the rule by adding one condition, coming up with a new rule (can be more than one according to number of attributes)
3. Evaluate the created rules using heuristics
4. Pruning: select only the best performing rule
5. repeat from step 2 until can no longer go any further

<img width="490" alt="Screenshot 2568-04-28 at 22 01 12" src="https://github.com/user-attachments/assets/26675294-df03-4102-9f7f-5ff1234224bd" />

$\text{SEQUENTIAL-COVERING} + \text{LEARN-ONE-RULE}$
- $Pos \leftarrow \text{ positive examples}$
- $Neg \leftarrow \text{ negative examples}$
- $while \  Pos \neq \\{\\} \ , do$
  - $Learn \  a \  NewRule$
    - $NewRule \leftarrow \text{ most general rule possible}$
    - $NewRuleNeg \leftarrow Neg$ (indicator of how many negative examples left)
    - $while \  NewRuleNeg \neq \\{\\}, do$
      - $add \  a  \  new \  literal \  to \  specialise \  NewRule$
        1. $Candidate\\_Literals \leftarrow \text{generate candidates}$
        2. $Best\\_literal \leftarrow \arg \max\_{L \in Candidate\\_literals} Performance (SpecialiseRule(NewRule, L))$
        3. $add \  Best\\_literal \  to \  NewRule \  preconditions$
        4. $NewRuleNeg \leftarrow subset \  of \  NewRuleNeg \  that \  satisfies \  NewRule \  preconditions$
    - $Learned\\_rules \leftarrow Learned\\_rules + NewRule$
    - $Pos \leftarrow Pos = \\{ \text{members of } Pos \text{ covered by } NewRule \\}$
- $Return \  Learned\\_rule$

## Subtleties: Learn One Rule

1. We can try *beam search* (the more beam the more cost, but not exponentially) &rarr; The algorithm explained earlier was *hill-climbing* algorithm.
2. Easily generalises to multi-valued target functions &rarr; still functionable even the non-binary case
3. Choose evaluation function to guide search:
  - Entropy
  - Sample accuracy

```math
\frac{n_c}{n}
```
  
  - $m$ estimate:

```math
\frac{n_c + mp}{n + m}
```

## Variant of Rule Learning Programs

- *Sequential* or *Simultaneous* covering of data?
- General &rarr; specific, or specific &rarr; general (initialise with most specific, then try create another slightly more general rule by removing one condition and see which covers more positive example)?
- Generate-and-test, or example-driven (try derive a rule from a positive example)?
- Whether and how to post-prune (we can try pruning resulting rules later)?
- What statistical evaluation function?

## Learning First Order Rules

Earlier, we talked in propositional rule, but sometime this cannot be done on programming language with only first order rule (rules with varables)

example:

$Ancestor(x,y) \leftarrow Parent(x,y)$  
$Ancestor(x,y) \leftarrow Parent(x,z) \land Ancestor(z,y)$

<img width="467" alt="Screenshot 2568-04-28 at 22 36 10" src="https://github.com/user-attachments/assets/fb0ce5fe-f103-42bf-b4a3-1e564dbbb2f6" />

To do this:

$\text{FOIL} (Target\\_predicate, Predicates, Examples)$
- $Pos \leftarrow \text{ positive examples}$
- $Neg \leftarrow \text{ negative examples}$
- $while \  Pos \neq \\{\\} \ , do$
  - $Learn \  a \  NewRule$
    - $NewRule \leftarrow \text{ most general rule possible}$
    - $NewRuleNeg \leftarrow Neg$ (indicator of how many negative examples left)
    - $while \  NewRuleNeg \neq \\{\\}, do$
      - $add \  a  \  new \  literal \  to \  specialise \  NewRule$
        1. $Candidate\\_Literals \leftarrow \text{generate candidates}$
        2. $Best\\_literal \leftarrow \arg \max\_{L \in Candidate\\_literals} Foil\\_Gain(L, NewRule)$
        3. $add \  Best\\_literal \  to \  NewRule \  preconditions$
        4. $NewRuleNeg \leftarrow subset \  of \  NewRuleNeg \  that \  satisfies \  NewRule \  preconditions$
    - $Learned\\_rules \leftarrow Learned\\_rules + NewRule$
    - $Pos \leftarrow Pos = \\{ \text{members of } Pos \text{ covered by } NewRule \\}$
- $Return \  Learned\\_rule$

### Specialising in FOIL

Learning rule: $P(x_1, x_2, \dots, x_k) \leftarrow L_1 \dots L_n$

Candidate specialisation add new literal of form:

- $Q(v_1, \dots, v_r)$ &larr; background knowledge; all possible predicates. At least one of the $v_i$ in the created literal must already exist as a variable in the rule (the find relation).
- $Equal(x_j, x_k)$ where $x_j$ and $x_k$ are variables already present in the rule
- The negation of either of the above forms of literals

### Information gain in FOIL

```main
Foil\\_Gain(L,R) \equiv t \left( log_2 \frac{p_1}{p_1 + n_1} - log_2 \frac{p_0}{p_0 + n_0}
```

Where
- $L$ is the candidate literal to add to rule $R$
- $p_0$ = number of positive examples of $R$
- $n_0$ = number of negative examples of $R$
- $p_1$ = number of positive examples of $R+L$
- $n_1$ = number of negative examples of $R+L$
- $t$ is the number of positive bindings of $R$ also covered by $R+L$ (to weight the number so that the algorithm chooses the literal that covers more positive examples while ratio maybe equal)

Note:
- $\log_2 \frac{p_0}{p_0 + n_0}$ is optimal number of bits to indicate the class of a positive binding covered by $R$

<img width="313" alt="Screenshot 2568-04-29 at 02 12 29" src="https://github.com/user-attachments/assets/0797bbff-d40a-429f-b7cb-526a7a2cb64c" />

## Induction as Inverted Deduction

Induction is finding $h$ such that

```math
(\forall \langle x_i, f(x_i) \rangle \in D) B \land h \land x_i \vdash f(x_i)
```

> $\vdash$ &rarr; imply

where
- $x_i$ is $i$th training instance
- $f(x_i)$ is the target function value for $x_i$
- $B$ is other background knowledge

"pairs of people, $\langle u, v \rangle$ such that child of $u$ is $v$"

```math
f(x_i): Child(Bob, Sharon)
```

```math
x_i: Male(Bob), Female(Sharon), Father(Sharon, Bob)
```

```math
B: Parent(u, v) \leftarrow Father(u, v)
```

What satisfies $(\forall \langle x_i, f(x_i) \rangle \in D) B \land h \land x_i \vdash f(x_i)$ ?

```math
h_1: Child(u,v) \leftarrow Father(v,u)
```

> This can be proven, as stated in the example. We can try this by assigning $Bob$ to $v$ and $Sharon$ to $u$.

```math
h_2: Child(u,v) \leftarrow Parent(v,u)
```

> This too can be proven. Although a bit better in practice, as being a parent does not specify sex. &rarr; more generalised.

