"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x, y):
    return x * y

def id(x):
    return x

def add(x, y):
    return float(x + y)

def neg(x):
    return float(-x)

def lt(x, y):
    return float(x < y)

def eq(x, y):
    return float(x == y)

def max(x, y):
    return x if x >= y else y

def is_close(x, y):
    return abs(x - y) < 1e-2

def sigmoid(x):
    """Calculate the sigmoid function."""
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))

def relu(x):
    """Apply the ReLU activation function."""
    return x if x > 0 else 0.0

def log(x):
    """Calculate the natural logarithm."""
    if x <= 0:
        raise ValueError("log(x): x must be positive")
    return math.log(x)

def exp(x):
    """Calculate the exponential function."""
    return math.exp(x)

def inv(x):
    """Calculate the reciprocal."""
    if x == 0:
        raise ZeroDivisionError("inv(x): x cannot be zero")
    return 1 / x

def log_back(x, d):
    """Compute derivative of log(x) times upstream gradient d."""
    if x <= 0:
        raise ValueError("log_back(x, d): x must be positive")
    return d / x

def inv_back(x, d):
    """Compute derivative of 1/x times upstream gradient d."""
    if x == 0:
        raise ZeroDivisionError("inv_back(x, d): x cannot be zero")
    return -d / (x ** 2)

def relu_back(x, d):
    """Compute derivative of ReLU(x) times upstream gradient d."""
    return d if x > 0 else 0

def sigmoid_back(x, d):
    """Compute derivative of sigmoid(x) times upstream gradient d."""
    s = sigmoid(x)
    return d * s * (1 - s)

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.

def map(fn):
    """
    Higher-order map function.
    Returns a function that applies `fn` to each element of an iterable.
    """
    def apply(xs):
        return [fn(x) for x in xs]
    return apply


def zipWith(fn):
    """
    Higher-order zipWith function.
    Returns a function that applies `fn` to pairs of elements from two iterables.
    """
    def apply(xs, ys):
        return [fn(x, y) for x, y in zip(xs, ys)]
    return apply


def reduce(fn, start):
    """
    Higher-order reduce function.
    Reduces an iterable to a single value by applying `fn`.
    """
    def apply(xs):
        result = start
        for x in xs:
            result = fn(result, x)
        return result
    return apply

negList = map(neg)

addLists = zipWith(add)

sum = reduce(add, 0)

prod = reduce(mul, 1)