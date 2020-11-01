# Optimization algorithm
<!-- vim-markdown-toc GFM -->

* [Gradient Decent variants](#gradient-decent-variants)
    * [Gradient Decent](#gradient-decent)
    * [mini-batch Gradient Decent](#mini-batch-gradient-decent)
    * [Stochastic Gradient Descent](#stochastic-gradient-descent)
* [Challenges](#challenges)
* [Momentum](#momentum)
* [Nesterov accelerated gradient(NAG)](#nesterov-accelerated-gradientnag)
* [Referrences](#referrences)

<!-- vim-markdown-toc -->

## Gradient Decent variants
```python
# Vanilla update
x += - learning_rate * dx
```
### Gradient Decent
All sample
### mini-batch Gradient Decent
Partition sample
### Stochastic Gradient Descent
Single sample
## Challenges
1. Choosing a proper learning rate can be difficult
2. Learning rate schedules and thresholds have to be defined in advance and are
   thus unable to adapt to a dataset's charateristics.
3. The same learning rate applies to all parameter updates.
4. Getting trapped in the suboptimal local minima or saddle points.

## Momentum
```python
# Momentum update
v = mu * v - learning_rate * dx # integrate velocity
x += v # integrate position
```
## Nesterov accelerated gradient(NAG)
```python
x_ahead = x + mu * v
# evaluate dx_ahead (the gradient at x_ahead instead of at x)
v = mu * v - learning_rate * dx_ahead
x += v
```
## Referrences
1. [An overview of gradient descent optimization algorithms.](http://ruder.io/optimizing-gradient-descent/index.html)
