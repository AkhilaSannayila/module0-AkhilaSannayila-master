"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
# Implementation of a prelude of elementary functions.

# -------------------------------------------
# ✅ Task 0.1: Basic Mathematical Operators
# -------------------------------------------

def mul(x: float, y: float) -> float:
    """
    Multiply two numbers.
    
    Args:
        x: First number.
        y: Second number.
    
    Returns:
        Product of x and y.
    """
    return x * y


def id(x: float) -> float:
    """
    Return the input unchanged.
    
    Args:
        x: Input value.
    
    Returns:
        Same value as x.
    """
    return x


def add(x: float, y: float) -> float:
    """
    Add two numbers.
    
    Args:
        x: First number.
        y: Second number.
    
    Returns:
        Sum of x and y.
    """
    return x + y


def neg(x: float) -> float:
    """
    Negate a number.
    
    Args:
        x: Input value.
    
    Returns:
        Negative of x.
    """
    return -x


def lt(x: float, y: float) -> float:
    """
    Check if x is less than y.
    
    Args:
        x: First value.
        y: Second value.
    
    Returns:
        1.0 if x < y, else 0.0.
    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """
    Check if two numbers are equal.
    
    Args:
        x: First value.
        y: Second value.
    
    Returns:
        1.0 if x == y, else 0.0.
    """
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """
    Return the larger of two numbers.
    
    Args:
        x: First value.
        y: Second value.
    
    Returns:
        Larger of x and y.
    """
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """
    Check if two numbers are close within a small tolerance.
    
    Args:
        x: First value.
        y: Second value.
    
    Returns:
        1.0 if |x - y| < 1e-2, else 0.0.
    """
    return float(abs(x - y) < 1e-2)


def sigmoid(x: float) -> float:
    """
    Sigmoid activation function.
    
    Args:
        x: Input value.
    
    Returns:
        Sigmoid of x.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1.0 + exp_x)


def relu(x: float) -> float:
    """
    ReLU activation function.
    
    Args:
        x: Input value.
    
    Returns:
        x if x > 0, else 0.
    """
    return max(x, 0)

# Small constant for numerical stability
EPS = 1e-6

def log(x: float) -> float:
    """
    Natural logarithm with numerical stability.
    
    Args:
        x: Input value.
    
    Returns:
        Natural logarithm of x.
    """
    return math.log(x + EPS)


def exp(x: float) -> float:
    """
    Exponential function.
    
    Args:
        x: Input value.
    
    Returns:
        e^x.
    """
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """
    Derivative of log function times a second argument.
    
    Args:
        x: Input value.
        d: Second argument.
    
    Returns:
        d * (1 / x)
    """
    return d * 1 / x


def inv(x: float) -> float:
    """
    Reciprocal of a number.
    
    Args:
        x: Input value.
    
    Returns:
        1 / x.
    """
    if x == 0:
        raise ValueError("Cannot divide by zero")
    return 1 / x


def inv_back(x: float, d: float) -> float:
    """
    Derivative of reciprocal function times a second argument.
    
    Args:
        x: Input value.
        d: Second argument.
    
    Returns:
        -d / (x ** 2)
    """
    return -d / (x ** 2)


def relu_back(x: float, d: float) -> float:
    """
    Derivative of ReLU function times a second argument.
    
    Args:
        x: Input value.
        d: Second argument.
    
    Returns:
        d if x > 0, else 0.
    """
    return d if x > 0 else 0


# -------------------------------------------
# ✅ Task 0.3: Higher-Order Functions
# -------------------------------------------

def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """
    Higher-order map function.
    
    Args:
        fn: Function that takes a float and returns a float.
    
    Returns:
        A function that applies `fn` to each element of a list.
    """
    return lambda ls: [fn(x) for x in ls]


def negList(ls: Iterable[float]) -> Iterable[float]:
    """
    Negate each element in a list using map and neg.
    
    Args:
        ls: List of float values.
    
    Returns:
        New list with each element negated.
    """
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """
    Higher-order zipWith function.
    
    Args:
        fn: Function to combine two values.
    
    Returns:
        Function that takes two lists and applies fn to pairs of elements.
    """
    return lambda ls1, ls2: [fn(x, y) for x, y in zip(ls1, ls2)]


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """
    Add corresponding elements of two lists using zipWith and add.
    
    Args:
        ls1: First list.
        ls2: Second list.
    
    Returns:
        List with element-wise sum.
    """
    return zipWith(add)(ls1, ls2)


def reduce(fn: Callable[[float, float], float], start: float) -> Callable[[Iterable[float]], float]:
    """
    Higher-order reduce function.
    
    Args:
        fn: Function to combine two values.
        start: Initial value.
    
    Returns:
        Function that reduces a list to a single value.
    """
    def reduce_func(ls: Iterable[float]) -> float:
        acc = start
        for x in ls:
            acc = fn(acc, x)
        return acc
    
    return reduce_func


def sum(ls: Iterable[float]) -> float:
    """
    Sum up all elements in a list using reduce.
    
    Args:
        ls: List of float values.
    
    Returns:
        Sum of the elements.
    """
    return reduce(add, 0)(ls)


def prod(ls: Iterable[float]) -> float:
    """
    Product of all elements in a list using reduce.
    
    Args:
        ls: List of float values.
    
    Returns:
        Product of the elements.
    """
    return reduce(mul, 1)(ls)
