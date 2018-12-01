from datetime import datetime
import sys
import os
from . import tweaks


class Individual:
    """
    A convenient representation of an individual for the computation. This representation contains the genetic material
    and the score associated with this individual.
    """

    def __init__(self, gene, score=None, gen_id=None):
        self.gene = gene
        self.score = score
        self.gen_id = gen_id

    @staticmethod
    def get_scores(population):
        return map(lambda p: p.score, population)

    @staticmethod
    def get_genes(population):
        return map(lambda p: p.gene, population)

    def copy(self):
        return Individual(gene=dict(self.gene), score=self.score, gen_id=self.gen_id)


class Callback:
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def on_strategy_begin(self, logs=None):
        pass

    def on_gen_end(self, logs=None):
        pass


class History(Callback):
    def __init__(self):
        self.generations = []
        super().__init__()

    def on_strategy_begin(self, logs=None):
        self.generations = []

    def on_gen_end(self, logs=None):
        self.generations.append(logs)


class ConsoleLog(Callback):
    def __init__(self):
        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._prev_output_len = 0
        super().__init__()

    def _write_msg(self, msg):
        if self._dynamic_display:
            sys.stdout.write('\b' * self._prev_output_len)
            sys.stdout.write('\r')
            self._prev_output_len = len(msg)
            sys.stdout.write(msg)
        else:
            sys.stdout.write(msg + '\n')

    def on_gen_end(self, logs=None):
        if logs is None:
            logs = {}
        gen_id = logs.get('gen_idx', 'NA')
        gen_time = logs.get('gen_time', 'NA')
        best_score = logs.get('best_score', 'NA')
        population_mean_score = logs.get('population_mean_score', 'NA')
        prefix = '*' if logs.get('hit', False) else '-'

        msg = f'{prefix} gen_idx: {gen_id:{3}}, gen_time: {gen_time:{6}.{3}}, best_score: {best_score:{6}.{6}}, ' \
              f'pop_mean_score: {population_mean_score:{6}.{6}}'
        self._write_msg(msg)


class ModelCheckpoint(Callback):
    """
    A callback that saves the best model after every hit.
    """

    def __init__(self, path='.'):
        self.path = path
        super().__init__()

    def on_gen_end(self, logs=None):
        if logs is None or (not logs.get('hit', False)):
            return
        time = str(datetime.now().isoformat())
        file = os.path.join(self.path, f'model-{time}')
        with open(file, 'wb') as outs:
            tweaks.TweaksSerializer.save(self.model.best, outs)
