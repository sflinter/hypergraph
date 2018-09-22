from .cgp import Operators, FuncMark
import numpy as np
import math


class IntOperators(Operators):
    def __init__(self, k=64):
        self.k = int(k)
        super().__init__(input_count=2)

    @property
    def null_value(self):
        return 0

    def null_like(self, x):
        return 0

    @FuncMark('base')
    def op_const(self, x, y, p):
        return int(np.round(p*self.k))

    @staticmethod
    @FuncMark('base')
    def op_identity(x, y, p):
        return x

    @staticmethod
    @FuncMark('base')
    def op_ywire(x, y, p):
        return y

    @staticmethod
    @FuncMark('math')
    def op_add(x, y, p):
        return x + y

    @staticmethod
    @FuncMark('math')
    def op_mul(x, y, p):
        if x >= 65536 or y >= 65536:
            return 0
        return x * y

    #@staticmethod
    #@FuncMark('math')
    #def op_pow_2(x, y, p):
    #    return x ** 2

    #@staticmethod
    #@FuncMark('math')
    #def op_pow_3(x, y, p):
    #    return x ** 3

    @staticmethod
    @FuncMark('math')
    def op_div(x, y, p):
        if y == 0:
            return 0
        return x // y

    @staticmethod
    @FuncMark('math')
    def op_mod(x, y, p):
        if y == 0:
            return 0
        return x % y

    @staticmethod
    @FuncMark('math')
    def op_max(x, y, p):
        return max(x, y)

    @staticmethod
    @FuncMark('math')
    def op_min(x, y, p):
        return min(x, y)

    @staticmethod
    @FuncMark('math')
    def op_factorial(x, y, p):
        x = abs(x)
        if x >= 12:
            return 0
        return math.factorial(x)

    @staticmethod
    @FuncMark('math')
    def op_sign(x, y, p):
        if x > 0: return 1
        if x < 0: return -1
        return 0

    @staticmethod
    @FuncMark('math')
    def op_add_inverse(x, y, p):
        return -x

    @staticmethod
    @FuncMark('math')
    def op_gcd(x, y, p):
        return math.gcd(x, y)

    @staticmethod
    @FuncMark('math')
    def op_lcm(x, y, p):
        v = abs(x * y)
        if v == 0:
            return 0
        return v//math.gcd(x, y)

    @staticmethod
    @FuncMark('math')
    def op_abs(x, y, p):
        return abs(x)

    @staticmethod
    @FuncMark('math')
    def op_is_even(x, y, p):
        if x % 2 == 0:
            return 1
        return 0

    @staticmethod
    @FuncMark('math')
    def op_is_odd(x, y, p):
        if x % 2 != 0:
            return 1
        return 0
