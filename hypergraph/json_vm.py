import numpy as np
import ctypes
import random


def hexfloat(s):
    """
    Converts a hex string to float
    :param s:
    :return:
    """
    if not isinstance(s, str):
        raise ValueError()

    i = int(s, 16)
    cp = ctypes.pointer(ctypes.c_int(i))
    fp = ctypes.cast(cp, ctypes.POINTER(ctypes.c_float))
    return fp.contents.value


class StdOps:
    #TODO: definitions, repeat, map, reduce, filter, rnd.shuffle

    def __init__(self, prefix):
        self.prefix = prefix

    def _op_last(self, list_):
        if list_ is None:
            return None
        return list_[-1]

    def _op_first(self, list_):
        if list_ is None:
            return None
        return list_[0]

    def _rnd_shuffle(self, list_):
        return random.shuffle(list_)

    def _op_list(self, list_):
        return list_

    def build_(self):
        p = self.prefix

        output = {
            # 'list' is useful for ['#$#list', '#$#none', 1, 2] => [None, 1, 2] so that the first element
            # can be expanded but not invoked as function
            p+'list': self._op_list,

            p+'last': self._op_last,
            p+'first': self._op_first,
            p+'none': None,
            p+'str': lambda x: str(x),
            p+'rnd.shuffle': self._rnd_shuffle,
            p+'hf': hexfloat,
            p+'rat': lambda a, b: a / b
        }

        return output

    @staticmethod
    def build(prefix):
        return StdOps(prefix).build_()


class VM:
    def __init__(self, ops_prefix='#$#', ops_defs={}):
        if not isinstance(ops_prefix, str):
            raise ValueError()
        if not isinstance(ops_defs, dict):
            raise ValueError()
        self.ops_prefix = ops_prefix
        self.ops_defs = StdOps.build(ops_prefix)
        self.ops_defs.update(ops_defs)

    def run(self, args):
        r = self.run

        if isinstance(args, dict):
            return dict([map(r, kv) for kv in args.items()])

        if isinstance(args, tuple):
            ret = r(list(args))
            if isinstance(ret, list):
                return tuple(ret)
            return ret

        if isinstance(args, list):
            if len(args) == 0:
                return args

            args = [r(obj) for obj in args]

            #TODO substitute all occurrences of #$#... before and then execute
            first = args[0]
            if isinstance(first, str) and first.startswith(self.ops_prefix):
                op = self.ops_defs.get(first)
                if op is not None:
                    return op(*args[1:])

        return args

    #TODO def __call__(self, *args, **kwargs):


def vm_test(prog):
    ops_prefix='#$#'
    ops_defs={'#$#sum': np.sum, '#$#square': np.square, '#$#np.ones': np.ones,
              '#$#rat': lambda a, b: a/b}
    vm = VM(ops_prefix, ops_defs)
    return vm.run(prog)


#prog = {'a': ('#$#rat', 1, 2)}
#print (vm_test(prog))
