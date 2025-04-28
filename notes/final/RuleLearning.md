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
        2. $Best\\_literal \leftarrow \arg \max\_{L \in Candidate\\_literals}$
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

$\text{FOIL}$

