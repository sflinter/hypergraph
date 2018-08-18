from hp_utils import TypeMisMatchError


class StateContainer:
    def __init__(self):
        self.states = []

    def add_state(self, state):
        if isinstance(state, State):
            self.states.append(state)
        else:
            raise TypeMisMatchError(State(), state)

    def get_states(self):
        return self.states


class State:
    def __init__(self):
        self.params = []

    def add_param(self):
        pass


class Parameter:
    pass

sc = StateContainer()



state= 'a'
sc.add_state(state)