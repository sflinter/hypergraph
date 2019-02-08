# A series of adapters to access some important algorithms of the notorious hyperopt, a well-established library
# for hyper-parameters optimization.

from hyperopt import hp
from . import tweaks as Tweaks
from .utils import HGError


def _uniform(key, tweak: Tweaks.Uniform):
    if tweak.size is not None:
        raise HGError('Uniform distribution with non-None size not supported')
    r = tweak.range
    return hp.uniform(key, r[0], r[1])


def _quniform(key, tweak: Tweaks.QUniform):
    if tweak.size is not None:
        raise HGError('QUniform distribution with non-None size not supported')
    r = tweak.range
    return hp.quniform(key, r[0], r[1], tweak.q)


def _uniform_perm(key, tweak: Tweaks.UniformPermutation):
    n = len(tweak.values)
    return {    # TODO use a namedtuple
        'indexes': [hp.randint(str(key)+'-perm-'+str(i), n-i) for i in range(tweak.k)],
        'values': list(tweak.values)
    }


def _normal(key, tweak: Tweaks.Normal):
    if tweak.size is not None:
        raise HGError('Normal distribution with non-None size not supported')
    return hp.normal(key, tweak.mu, tweak.sigma)


_adapters = {
    Tweaks.UniformChoice: lambda k, t: hp.choice(k, t.values),
    Tweaks.Uniform: _uniform,
    Tweaks.QUniform: _quniform,
    Tweaks.Normal: _normal,
    Tweaks.UniformPermutation: _uniform_perm
}

_PERM_ADAPTERS_KEY = '__hg2hpopt_perm_adapters__'


def tweaks2hpopt(tweaks: dict):
    """
    Adapt the tweaks based on class tweaks.Distribution to Hyperopt. Not all distributions present in this framework
    are supported by Hyperopt so we overcome to this limitation by creating some temporary structures that
    Hyperopt can manage.
    :param tweaks: A dictionary of tweak configs.
    :return: A new dictionary of tweaks that Hyperopt can manage.
    """
    output_hp = {}
    perm_adapters_keys = []
    for key, value in tweaks.items():
        if isinstance(value, Tweaks.Distribution):
            try:
                adapter = _adapters[type(value)]
            except KeyError:
                raise HGError('Tweak not supported')
            if isinstance(value, Tweaks.UniformPermutation):
                perm_adapters_keys.append(key)
            output_hp[key] = adapter(key, value)
        else:
            output_hp[key] = value
    if len(perm_adapters_keys):
        output_hp[_PERM_ADAPTERS_KEY] = perm_adapters_keys
    return output_hp


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


def expand_hpopt_adapters(params: dict):
    """
    Expand tweaks adapters after Hyperopt processing
    :param params: The tweaks processed by hyperopt
    :return: A new tweaks dictionary with the internal Hyperopt tricks expanded
    """
    perm_adapters_keys = params.get(_PERM_ADAPTERS_KEY)
    output = dict(params)
    if perm_adapters_keys is not None:
        del output[_PERM_ADAPTERS_KEY]
        for key in perm_adapters_keys:
            output[key] = _expand_perm(output[key])     # substitute with the expanded permutation
    return output
