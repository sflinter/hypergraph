# A series of adapters to access some important algorithms of the notorious hyperopt, a well-established library
# for hyper-parameters optimization.

import hyperopt as hp
from . import tweaks as Tweaks
from .utils import HGError


def _uniform(key, tweak: Tweaks.Uniform):
    if tweak.size is not None:
        raise HGError('Uniform distribution with non-None size not supported')
    r = tweak.range
    return hp.uniform(key, r[0], r[1])


def _uniform_perm(key, tweak: Tweaks.UniformPermutation):
    n = len(tweak.values)
    return {
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
    Tweaks.Normal: _normal,
    Tweaks.UniformPermutation: _uniform_perm
}

_PERM_ADAPTERS_KEY = '__hg2hpopt_perm_adapters__'


def tweaks2hpopt(tweaks: dict):
    """
    Adapt the tweaks based on class tweaks.Distribution to Hyperopt
    :param tweaks:
    :return:
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
        output_hp[key] = value
    if len(perm_adapters_keys):
        output_hp[_PERM_ADAPTERS_KEY] = perm_adapters_keys
    return output_hp


def expand_hpopt_adapters(tweaks: dict):
    """
    Expand tweaks adapters after hypeopt processing
    :param tweaks:
    :return:
    """
    perm_adapters_keys = tweaks.get(_PERM_ADAPTERS_KEY)
    output = dict(tweaks)
    if perm_adapters_keys is not None:
        del output[_PERM_ADAPTERS_KEY]
        # TODO
        raise NotImplemented()
    return output
