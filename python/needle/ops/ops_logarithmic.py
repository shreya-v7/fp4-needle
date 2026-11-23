from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z = array_api.max(Z, axis=1, keepdims=True)
        shifted_z = Z - max_z

        log_sum_exp = array_api.log(
            array_api.sum(
                array_api.exp(shifted_z),
                axis=1,
                keepdims=True,
            )
        )

        return shifted_z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        summed_grad = summation(out_grad, axes=(1,))
        summed_grad = reshape(summed_grad, (out_grad.shape[0], 1))
        summed_grad = broadcast_to(summed_grad, out_grad.shape)

        return out_grad - exp(node) * summed_grad
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # changed for NDArray backend
        max_z = Z.max(axis=self.axes, keepdims=True)
        if self.axes is None:
            max_z = max_z.reshape((1,) * len(Z.shape))

        shifted_z = Z - max_z.broadcast_to(Z.shape)

        return (
            array_api.log(
                array_api.sum(
                    array_api.exp(shifted_z),
                    axis=self.axes
                )
            )
            + Z.max(axis=self.axes)
        )
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]

        if self.axes is None:
            axes = tuple(range(len(Z.shape)))
        # changed: also accept an int axis
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = self.axes

        reshape_shape = list(Z.shape)

        for axis in axes:
            reshape_shape[axis] = 1

        reshape_shape = tuple(reshape_shape)

        lse = reshape(node, reshape_shape)
        lse = broadcast_to(lse, Z.shape)

        grad = reshape(out_grad, reshape_shape)
        grad = broadcast_to(grad, Z.shape)

        return grad * exp(Z - lse)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

