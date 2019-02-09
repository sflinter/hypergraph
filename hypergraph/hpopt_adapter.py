# A series of adapters to access some important algorithms of the notorious hyperopt, a well-established library
# for hyper-parameters optimization.

from hyperopt import hp
from . import tweaks as Tweaks
from .utils import HGError

# TODO bug fix, native QUniform tweaks produce int whereas hp produces float, see the Switch tweak, a int(choice)
# has been temporarily put there


def _uniform(key, tweak: Tweaks.Uniform):
    if tweak.size is not None:
        raise HGError('Uniform distribution with non-None size not supported')
    r = tweak.range
    return hp.uniform(key, r[0], r[1]), None


def _expand_to_int(value):
    return int(value)


def _quniform(key, tweak: Tweaks.QUniform):
    if tweak.size is not None:
        raise HGError('QUniform distribution with non-None size not supported')
    r = tweak.range
    return hp.quniform(key, r[0], r[1], tweak.q), _expand_to_int


def _expand_perm(desc):
    """
    Expand the permutation, the parameter is a descriptor of the form:
    {
        'indexes': [randint(n), randint(n-1), ...],
        'values': [values...]
    }
    :param desc:
    :return: A list of selected values according to the permutation indexes
    """
    values = list(desc['values'])
    output = list()
    for idx in desc['indexes']:
        output.append(values[idx])
        del values[idx]
    return output


def _uniform_perm(key, tweak: Tweaks.UniformPermutation):
    n = len(tweak.values)
    return {    # TODO use a namedtuple
        'indexes': [hp.randint(str(key)+'-perm-'+str(i), n-i) for i in range(tweak.k)],
        'values': list(tweak.values)
    }, _expand_perm


def _normal(key, tweak: Tweaks.Normal):
    if tweak.size is not None:
        raise HGError('Normal distribution with non-None size not supported')
    return hp.normal(key, tweak.mu, tweak.sigma), None


def _choice(key, tweak: Tweaks.UniformChoice):
    return hp.choice(key, tweak.values), None


_adapters = {
    Tweaks.UniformChoice: _choice,
    Tweaks.Uniform: _uniform,
    Tweaks.QUniform: _quniform,
    Tweaks.Normal: _normal,
    Tweaks.UniformPermutation: _uniform_perm
}

_HPOPT_ADAPTERS_KEY = '__hg2hpopt_adapters__'


def tweaks2hpopt(tweaks: dict):
    """
    Adapt the tweaks based on class tweaks.Distribution to Hyperopt. Not all distributions present in this framework
    are supported by Hyperopt so we overcome to this limitation by creating some temporary structures that
    Hyperopt can manage.
    :param tweaks: A dictionary of tweak configs.
    :return: A new dictionary of tweaks that Hyperopt can manage.
    """
    output_hp = {}
    expanders = []  # contains tuples of the form (<key>, <expander function>)
    # expanders are counter-adapters to convert the tweaks back to hypergraph compatible format
    for key, value in tweaks.items():
        if isinstance(value, Tweaks.Distribution):
            try:
                adapter = _adapters[type(value)]
            except KeyError:
                raise HGError('Tweak not supported')
            # if isinstance(value, Tweaks.UniformPermutation):
            #    perm_adapters_keys.append(key)
            output_hp[key], expander = adapter(key, value)
            if expander is not None:
                expanders.append((key, expander))
        else:
            output_hp[key] = value
    output_hp[_HPOPT_ADAPTERS_KEY] = expanders
    return output_hp


def expand_hpopt_adapters(params: dict):
    """
    Expand tweaks adapters after Hyperopt processing
    :param params: The tweaks processed by hyperopt
    :return: A new tweaks dictionary with the internal Hyperopt tricks expanded
    """
    expanders = params.get(_HPOPT_ADAPTERS_KEY)
    output = dict(params)
    if expanders is not None:
        del output[_HPOPT_ADAPTERS_KEY]
        for key, expander in expanders:
            output[key] = expander(output[key])
    return output
