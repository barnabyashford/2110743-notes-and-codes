# 2110743 Machine Learning Midterm Note

## Week 01: Introduction to ML

### Why ML?

- It has been proven in many cases that <mark>**ML simply works!**</mark>

- There are many learning algorithms (e.g., learning from theory, rule learning), but one stands out more than others: <mark>**Learning from examples**</mark>

- <mark>**Growing online data**</mark> provides even more examples, aiding the popularity in learning-from-examples algorithm.

- ML were not popular before for its algorithm took a huge amount of time for hardware at the time to complete learning, but now, with immensly <mark>**more powerful computational power available**</mark>, ML now takes a matter of seconds.

- ML used to be a matter of interest only in academic world. Now <mark>**industries are investing in this particular field as well**</mark>, making the knowledge expands even more rapidly.

### Three niches for ML

1. <mark>**Data Mining**</mark>: sees a pattern or significance from a big pile of historical data.
  - *Ex.*: 
    - medical records &rarr; medical knowledge
2. <mark>**Software applications impossible for programming by hand**</mark>
  - *Ex.*: 
    - auto-pilot driving
    - speech recognition
3. <mark>**Self customising programs**</mark>
  - *Ex.*: 
    - user-personalising softwares such as newsreader that learn user interests

**Note:** These three examples are not the only ways ML can be implemented, but are the most frequent and popular.

#### Data Mining

##### Example

**Data:**

<img width="612" alt="Screenshot 2568-03-01 at 17 17 42" src="https://github.com/user-attachments/assets/4beb79f3-b985-4132-9641-81b5d33efd28" />

> Each specific aspect of data can be called as 'attribute' or 'feature'. In this case, the attributes/features are `Age`, `FirstPregnancy`, `Anemia`, `Diabetes`, `PreviousPrematureBirth`, `Ultrasound`, `Elective C-Section`, and `Emergency C-Section`. Note that sometimes the attributes can be 'unknown' as well, because it is natural for data to contain noises or missing attributes.

> The dimensions of data or attributes that we want the ML model to learn to 'predict' are called 'target attribute' or simple 'target'. In this case, the target attibute is `Emergency C-Section`.

**Given:**

- 9714 patient records
- Each patient record contains 215 features (shown in the image are merely examples)

**Learn to predict:**

- Classes of future patients at high risk for Emergency Cesarean Section (Emergency C-Section)

##### Result

<img width="612" alt="Screenshot 2568-03-01 at 17 17 42" src="https://github.com/user-attachments/assets/4beb79f3-b985-4132-9641-81b5d33efd28" />

One of 18 learned rules:

> This example implemented 'rule learning algorithm' that learns 'rule' from examples.

```
If No previous vaginal delivery, and Abnormal 2nd Trimester Ultrasound, and Malpresentation at admission
Then Probability of Emergency C-Section is 0.6

Over training data: 26/41 = .63,
Over test data: 12/20 = 60
```

In ML model training, we normally separate data into 'training' and 'test' data. Training data is to be used for model training, while test data is used to evaluate the performance of the model we have trained.

**Remember, never ever invole test data in training phase**

In more recent days of ML, there is also a phase where we choose different settings for ML model, a.k.a. 'parameter tuning'. Now we separate data into three parts:
- Training phase:
  - real training: training data
  - parameter tuning: validation data
- Test phase: test data

##### Some more examples

<img width="620" alt="Screenshot 2568-03-01 at 17 43 47" src="https://github.com/user-attachments/assets/b8930760-c8e8-4343-8867-13627cfcae34" />

<img width="572" alt="Screenshot 2568-03-01 at 17 44 06" src="https://github.com/user-attachments/assets/f8c959f7-a785-4e1b-8d8c-4882dedb9840" />

#### Software applications impossible for programming by hand

Problem: ALVINN \[Pomerleau\] drives 70 mph on highways

<img width="434" alt="Screenshot 2568-03-01 at 17 47 49" src="https://github.com/user-attachments/assets/d35e60dd-4abb-494a-9f52-6552af87f6c8" />

#### Self customising programs

Example: www.westwire.com

<img width="349" alt="Screenshot 2568-03-01 at 17 48 50" src="https://github.com/user-attachments/assets/8eb09b29-b47e-4cc5-b9bb-e32814dfb915" />

### Where is ML headed?

What can we do now?

- detailed learning of problems using neural network (Although it's really accurate, it's hard to interpret what's happening inside the model.)

- learning from unstructured data (e.g., free text, voice)! (ML used to require a well-structured database, now we can automate cleaning!)

- learn across mutiple database, getting data from the internet

- learn by active experimentation: now the models can tell us what else do they require for learning

- ML models are no longer limited to 'prediction', but now 'decision'

- ML models can now keep learning from more data

- syntax correction in programming language

### Relevant Disciplines

- AI (ML is a subfield in AI, *they are not the same*.)
  - Note that now ML is so advanced that they play significant roles in many other discipline, for example, Natural Language Processing (NLP).
- Bayesian methods (Probability)
- Computational complexity theory
- Control theory
- Information theory
- and more!

### Learning Problem

**Learning**: improving one's performance from experience at some tasks

Varibles:
- $T$: improve over **T**ask
- $P$: with respect to **P**erformance measure
- $E$: base on **E**xperience (examples)

Example: Playing checkers
- $T$: play checkers
- $P$: % of games won on in world tournament
- $E$: opportunity to play against self (or human player)

#### Learning to play checkers

- $T$: play checkers
- $P$: % of games won on in world tournament
- What experience?
- What should be learned?
- How shall it be represented? (How do we represent the knowledge learned?)
- What algorithm to be used?

##### Type of Training Experience (Examples)

- Direct or indirect?
  - Direct: telling explicitly if a step the model takes is good or not
  - Indirect: let the model take all the steps until finish and evaluate the entire result
 
- Teacher or not? (supervise vs unsupervised)

**consider this**: is the training example representative of the objective of learning?
