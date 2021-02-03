% Lecture 2: Machine Learning Basics
% CS 182 Spring 2021 -- Taught by Sergey Levine
% Notes by Michael Zhu

# Types of Machine Learning Problems

1. **Supervised Learning**

- have dataset of labeled data
- want to predict y from x

2. **Unsupervised learning**

- have unlabeled data
- want to learn a useful representation of the data

3. **Reinforcement learning**

- have an agent and environment
- agent outputs action and the environment outputs the next state and reward
- want to learn an optimal action strategy

# Supervised Learning

Given a labeled dataset $D = \{(x_{1}, y_{1}), (x_{2}, y_{2}), ..., (x_{n}, y_{n})\}$

Learn $f_{\theta}(x) = y$

## Considerations:

**How do we represent $f_{\theta}(x)$?**

This course goes over neural network representations of $f_{\theta}(x)$

**How do we measure the performance of $f_{\theta}(x)$?**

We compute the _loss_ which is usually a function of the difference between $f_{\theta}(x_{i})$ and $y_{i}$

**How do we find the best $\theta$?**

We use an optimzation algorithm. Examples include random search, local search, gradient descent, etc.

## Examples of supervised learning problems

Note: majority of machine learning in industry is supervised learning

| **Predict**         | **Based on...**     |
| ------------------- | ------------------- |
| category of object  | image               |
| sentence in French  | sentence in English |
| presense of disease | X-ray image         |
| text phrase         | audio utterance     |

## Predicting probabilities

Often more useful to predict a **probability distribution** of the set of possible outputs $\{y_{i}\} \ \forall i$

This is easier to learn due to **smoothness**.

**Probabilities**: It's possible to update probability distribution by small amounts

**Discrete Labels**: By definition, it's impossible to update a discrete label by small amounts; you either change labels or you don't

Previous goal: ~~learn $f_{\theta}(x) = y$~~

New goal: **learn $f_{\theta}(x) = p_{\theta}(y|x)$**

Conditions:

- probabilties must be positive $\rightarrow \forall i \ p_{\theta} (y_{i}|x) > 0$
- probabilites must sum to one $\rightarrow \sum_{i} p_{\theta}(y_{i}|x) = 1$

### Softmax function

Any function that takes some input vector and outputs a probability vector that is **positive** and **sums to one**.

Ideally this is a one to one & onto function to prevent information loss

**Ways to make a number $z$ positive**: $z^2 \ \ |z| \ \ max(0,z) \ \ exp(z)$

**Note**: $exp(z)$ is one to one & onto; it maps the set of real numbers to the set of positive reals $R \rightarrow R^{+}$

**Normalization**: forces numbers to sum to one by dividing each number by the total sum $z_{i}/\sum_{i} z_{i}$

Use $exp(x)$ and normalization to define our softmax

$$p(y_{i}|x) = softmax_{i}(f_{\theta}(x)) = \frac{exp(f_{\theta, i}(x))}{\sum_{j} exp(f_{\theta, j}(x))}$$

There are many ways to get positive numbers that some to one, but the softmax is very commmonly used

# Unsupervised Learning

Given unlabeled data $D = \{x_{1}, x_{2}, ..., x_{n}\}$

Learn a **representation** of the data

Another way of thinking about this is that the data comes from some **continuous probability distribution**, and the goal of unsupervised learning is to help us figure out and _model_ that distribution

## Two formulations of unsupervised learning

### Generative Modeling

Neural network that generates outputs that are similar to the data that it was trained on (i.e. GANs, VAEs, pixel RNN, etc).

Example: Generating faces; the model must come up with an internal representation of what makes up a face

### Self-supervised Representation Learning

Task: remove parts of the data and try to predict the missing parts

The task itself is not that important, but it forces the model to learn useful representations that can then be used for other tasks

Example: BERT language model

- dataset of sentences
- train model by removing words and then predicting the missing words
- model uses word embeddings to make these predictions, and as it trains, it updates the embeddings to represent the corpus better
- these embeddings can then be used for other language tasks like machine translation and sentiment analysis

## Reinforcement Learning

Given:

1. Agent:

- takes in **state** $s_{t}$ as input
- outputs an **action** $a_{t}$

2. Environment:

- takes in **action** $a_{t}$ as input
- outputs the next **state** $s_{t+1}$ and a corresponding **reward** $r_{t+1}$

**Want to _learn_ $f_{\theta}(s_{t}) = a_{t}$ which _maximizes_ total reward $\sum_{t=1}^{H} r(s_{t}, a_{t})$**

**Note**: Supervised learning can be thought of as a specific type of reinforcement learning where the reward is the negative loss

**Example: training a dog**

- action: muscle movement
- state/observations: sight, smell, sound, etc
- reward: food

**Example: training a robot to walk**

- action: motor current or torque
- state/observations: camera images or video feed
- reward: distance travelled

**Example: supply chain management**

- action: what to purchase, method of distribution, etc
- state/observations: inventory levels
- reward: profit

**Example: reccomendations (ad placement, youtube videos, healthcare treatment)**

- action: giving a recommendation
- state/observations: user history
- reward: if user accepts the recommendation
