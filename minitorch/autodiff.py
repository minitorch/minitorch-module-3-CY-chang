from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    val1, val2 = list(vals), list(vals)
    val1[arg] += epsilon
    val2[arg] -= epsilon
    return (f(*val1) - f(*val2)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # TODO: Implement for Task 1.4.
    visited = set()
    ord = []

    def dfs(node: Variable):
        if node.unique_id in visited:
            return
        visited.add(node.unique_id)
        if not node.is_constant() and node.parents:
            for parent_node in node.parents:
                dfs(parent_node)
        ord.append(node)
    dfs(variable)
    return ord


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    sorted_nodes = reversed(topological_sort(variable))
    grad_values = {}
    grad_values[variable.unique_id] = deriv
    for node in sorted_nodes:
        current_node_grad = grad_values.get(node.unique_id, 0.0)
        if node.is_leaf(): 
            continue
        if node.history is None: # Fix minitorch/tensor.py:352: AssertionError when history is None in task 2_4
            continue
        for parent_node, grad in node.chain_rule(current_node_grad):
            if parent_node.is_leaf():
                parent_node.accumulate_derivative(grad)
            else:
                grad_values[parent_node.unique_id] = grad_values.get(parent_node.unique_id, 0.0) + grad
                


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
