from . import json_vm
from hyperopt import hp
from abc import ABC, abstractmethod
import itertools
from contextlib import contextmanager
from functools import partial, reduce
import types
import numpy as np
import numbers

# TODO create a special node to get an item from a list output (or a key from a dict output)
# TODO aggregator node?
# TODO Variable, some nodes will accept variables in the config...
# TODO invocation failure interceptors... In some cases we may decide to continue the execution even though a node fails
# TODO round robin switch
# TODO think how to connect this to LOG with encapsulated contexts

# TODO "assign" to set a variable
# TODO parallel(), while(var), expression()
# TODO import ... import json definition from url or file
# TODO special node to access current hpopt config
# TODO declare more graph tweaks such as switch
# TODO pure nodes
# TODO zip()
# TODO tweaks declaration in the graph, they are like placeholders with a range of validity and a default value
# TODO generate tweaks automatically by using reflection! Use annotations to mark functions responsible for tweaks
# TODO define dependency that is, do not execute a node before the dependencies are fulfilled

# TODO generalize input binding:
# eg {'str': {1: node_id('abc')}}
# only node_id are expanded


def fq_ident(idents, sep='.') -> str:
    """
    Create a fully qualified identifier
    :param idents: list of identifiers or a string
    :param sep: the separator for the identifiers
    :return: the fully qualified identifier as string
    """
    if isinstance(idents, str):
        return idents
    return reduce(lambda s1, s2: s1 + sep + str(s2), idents)


class ExecutionContext:
    """
    The execution context stores the runtime values of a graph's instance
    """

    _default = None

    @classmethod
    def get_instance(cls, ctx):
        if ctx is not None:
            if not isinstance(ctx, ExecutionContext):
                raise ValueError('Param ctx must be of type ExecutionContext')
            return ctx
        return cls.get_default()

    @classmethod
    def get_default(cls):
        """
        Returns the execution context. If there is no default context then a new one is instantiated
        :return:
        """
        if cls._default is None:
            cls._default = ExecutionContext()
        return cls._default

    @classmethod
    def reset_default(cls):
        """
        Resets the default execution context
        :return:
        """
        cls._default = None

    @contextmanager
    def as_default(self):
        """
        Returns a context manager that sets this context as default
        :return: A context manager
        """
        prev_default = ExecutionContext._default
        try:
            ExecutionContext._default = self
            yield
        finally:
            ExecutionContext._default = prev_default

    # TODO store here: runtime lazy links and hpopt params

    def __init__(self):
        self._vars_per_graph = {}

    def set_var(self, graph=None, var=None, value=None):
        var = Graph.get_node_ext(var, graph)
        graph = var.parent
        if not isinstance(var, Variable):
            raise ValueError('The node var must be of type Variable')

        vars_ = self._vars_per_graph.setdefault(graph, {})
        var.check_type(value)
        vars_[var.namespace] = value

    def get_var_value(self, graph=None, var=None):
        var = Graph.get_node_ext(var, graph)
        graph = var.parent
        if not isinstance(var, Variable):
            raise ValueError('The node var must be of type Variable')

        vars_ = self._vars_per_graph.get(graph)
        if vars_ is None:
            return None
        return vars_[var.namespace]


class Node(ABC):
    _id_counter = itertools.count()

    def __init__(self, namespace=None, flags=()):   # TODO rename name?
        if namespace is None:
            namespace = 'node-'+str(next(self._id_counter))

        if not isinstance(namespace, str):
            raise ValueError('Node name must be a string')
        self._ns = namespace
        self._parent = None     # the graph that contains this node
        self._input_binding = None
        self._flags = flags

        #if self is not Graph.get_default():
        #    Graph.get_default().add_node(self)

    @staticmethod
    def get_name(node) -> str:
        if isinstance(node, Node):
            return node.namespace
        if not isinstance(node, str):
            raise ValueError('The input param must be either a string identifier of an object of type Node, '
                             'the current type is {}'.format(type(node)))
        return node

    @property
    def namespace(self) -> str:     # TODO rename name
        return self._ns

    @property
    def consumed(self):
        """
        Indicates whether this node is part of a graph
        :return:
        """
        return self._parent is not None

    @property
    def parent(self):
        """
        This property returns the graph that contains this node
        :return: a Graph
        """
        return self._parent

    @property
    def flags(self) -> tuple:
        return self._flags

    def get_input_binding(self, hpopt_config={}):
        return self._input_binding

    def set_input_binding(self, input_binding):
        """
        Setter used by the container to inject input bindings.
        It can be used to intercept the injection during the graph construction to check whether all requirements for
        the node are fulfilled.
        :param input_binding:
        :return:
        """
        self._input_binding = input_binding   #TODO deepcopy

    # TODO remove hpopt_config_ranges, create a more generic idea,
    # for example provide node's "tweaks" (switches) relative to a specific context
    # one of the contexes is hpopt. Note that input bindings may depends on a particular "tweak"
    # so, first we apply tweaks and then we build the final links. The tweaks may override
    # the links declared in the graph, the custom links are declared in the execution context.
    # When we return a tweak configuration we return a list or dictionary (or a more complex structure)
    # where the user value are substituted by placeholders. Placeholders like hyperopt may contain a range of validity

    @abstractmethod
    def get_hpopt_config_ranges(self) -> dict:
        """
        Return a dictionary with the hyperopt ranges. The keys must be fully qualified identifiers of the internal
        options. This should not be invoked for a non-consumed node, that is a node not added to a graph (this because
        some ranges may depend from input bindings).
        :return: A dictionary
        """
        return {}

    @abstractmethod
    def __call__(self, input, hpopt_config={}):
        """
        This methods builds the layers based on the input and hyper perameter optimizer params
        :param input: The input(s) to this node
        :param hpopt_config: A dictionary with the values choosed by the hyper parameters optimizer, the params that aren't
        specified will take the default value (from definition)
        :return:
        """
        pass

    def serialize(self):
        """
        Serializes the node into a json serializable format
        :return:
        """
        raise NotImplementedError()


class Identity(Node):
    """
    A simple node that returns the input when invoked without any modification.
    """
    def __init__(self, name=None):
        super(Identity, self).__init__(name)

    @staticmethod
    def deserializer(id=None):
        return Identity(id)

    def get_hpopt_config_ranges(self):
        return {}

    def __call__(self, input, hpopt_config={}):
        return input


def identity(name=None) -> Node:
    node = Identity(name)
    Graph.get_default().add_node(node)
    return node


def mark(name=None) -> Node:
    """
    Mark is a pseudonym of identity. The idea is to mark internal placeholders (from the graph point of view)
    :param name: The identifier of the identity node
    :return:
    """
    node = Identity(name)
    Graph.get_default().add_node(node)
    return node


class Placeholder(Node):
    """
    A node that represents the input of a graph
    """
    def __init__(self, key=None, match_all_inputs=False):
        self.key = key
        self.match_all_inputs = bool(match_all_inputs)
        super(Placeholder, self).__init__()

    def select_input(self, input):
        if self.match_all_inputs:
            return input
        return input[self.key]

    @staticmethod
    def deserializer(key=None, match_all_inputs=False):
        return Placeholder(key, match_all_inputs)

    def set_input_binding(self, input_binding):
        if input_binding is not None:
            raise ValueError('Input binding not supported for placeholders')

    def get_hpopt_config_ranges(self):
        return {}

    def __call__(self, input, hpopt_config={}):
        raise ValueError('Placeholders are not executable')


def placeholder(key=None, match_all_inputs=False):
    node = Placeholder(key, match_all_inputs)
    Graph.get_default().add_node(node)
    return node


def placeholder_all(key=None):
    #TODO generate just one node placeholder_all per graph
    return placeholder(key, True)


class Static(Node):
    # TODO declare pure node

    def __init__(self, name=None, value=None):
        self.value = value
        super(Static, self).__init__(name, flags=('p',))

    @staticmethod
    def deserializer(value=None):
        return Identity(name=None, value=value)

    def get_hpopt_config_ranges(self):
        return {}

    def __call__(self, input, hpopt_config={}):
        if input is not None:
            raise ValueError('Static nodes are not supposed to receive any input')
        return self.value


def static(value) -> Node:
    """
    Returns a node representing a static value.
    :param value:
    :return:
    """
    node = Static(name=None, value=value)
    Graph.get_default().add_node(node)
    return node


class Merge(Node):
    def __init__(self, name=None, mode='d'):
        if not(mode in ('l', 'd')):
            raise ValueError()
        self.mode = mode
        super(Merge, self).__init__(name)

    def get_hpopt_config_ranges(self):
        return {}

    @staticmethod
    def _dict_to_items(obj):
        if isinstance(obj, dict):
            return obj.items()
        return obj

    def __call__(self, input, hpopt_config={}):
        if isinstance(input, list):
            if self.mode == 'd':
                d = {}
                d.update(itertools.chain(*map(self._dict_to_items, input)))
                return d

            if self.mode == 'l':
                return itertools.chain(*input)

            raise RuntimeError()
        raise ValueError()


def merge(mode):
    node = Merge(name=None, mode=mode)
    Graph.get_default().add_node(node)
    return node


class GetKeys(Node):
    """
    Gets elements from lists or dictionaries
    """

    def __init__(self, name=None, keys=None, output_type=None):
        self.keys = keys
        self.output_type = output_type
        if not(output_type in [None, 'd']):
            raise ValueError()
        super(GetKeys, self).__init__(name)

    @staticmethod
    def deserializer(keys, output_type=None):
        return GetKeys(name=None, keys=keys, output_type=output_type)

    def get_hpopt_config_ranges(self):
        return {}

    def __call__(self, input, hpopt_config={}):
        if self.output_type == 'd':
            values = multi_iterable_map(lambda k: input[k], multi_iterable_get_values_as_list(self.keys))
            keys = multi_iterable_get_keys_as_list(self.keys)
            return dict(zip(keys, values))

        return multi_iterable_map(lambda k: input[k], self.keys)


def get_keys(keys, output_type=None):
    node = GetKeys(name=None, keys=keys, output_type=output_type)
    Graph.get_default().add_node(node)
    return node


def get_input_keys(keys, output_type=None):
    return link(get_keys(keys=keys, output_type=output_type), placeholder_all())


class Dummy(Node):
    def __init__(self, namespace=None):
        super(Dummy, self).__init__(namespace)

    @staticmethod
    def deserializer(id):
        return Dummy(id)

    def get_hpopt_config_ranges(self):
        h = HPOptHelpers(self.namespace)
        default = [
            h.def_choice('option1', tuple(['abc'+str(i) for i in range(4)])),
            h.def_choice('option2', tuple(['def'+str(i) for i in range(4)]))
        ]
        return dict(default)

    def __call__(self, input, hpopt_config={}):
        hpopt_helper = HPOptHelpers(self.namespace, hpopt_config)
        return hpopt_helper.build_choice('option1', input, lambda input_, config: {'input': input_, 'config': config})


def dummy(name=None):
    node = Dummy(name)
    Graph.get_default().add_node(node)
    return node


class Switch(Node):
    """
    A node that switches between multiple inputs
    """

    def __init__(self, namespace=None, default_choice=None):
        #TODO allow different probabilities for different inputs
        self.default_choice = default_choice
        super(Switch, self).__init__(namespace)

    @staticmethod
    def deserializer(id):
        return Switch(id)

    def get_hpopt_config_ranges(self):
        input_binding = self._input_binding
        if input_binding is None:
            return {}

        h = HPOptHelpers(self.namespace)
        if isinstance(input_binding, list):
            return dict([h.def_choice('options', list(range(len(input_binding))))])

        if isinstance(input_binding, list):
            return dict([h.def_choice('options', list(input_binding.keys()))])

        raise ValueError()

    def get_input_binding(self, hpopt_config={}):
        choice = HPOptHelpers(self.namespace, hpopt_config).get_choice('options', self.default_choice)
        if choice is None:
            return None
        return self._input_binding[choice]

    def __call__(self, input, hpopt_config={}):
        return input
        #if input is None:
        #    return None
        #hpopt_helper = HPOptHelpers(self.namespace, hpopt_config)
        #return hpopt_helper.build_choice('options', input,
        #                                 lambda input_, config: input_[config],
        #                                 default_choice=self.default_choice)


def switch(name=None, default_choice=None) -> Node:
    node = Switch(name, default_choice)
    Graph.get_default().add_node(node)
    return node


def link(node, input_bindings=None) -> [Node, None]:
    """
    Creates a (lazy) link in the current graph between the node and the nodes identified by the input_bindings
    :param node: A string identifier of the node or a Node object
    :param input_bindings:
    :return: node if node is an instance of Node
    """

    graph = Graph.get_default()

    node = Graph.expand(node)   # TODO run adapter here
    graph.add_node_if_obj(node)

    if input_bindings is not None:
        # early checking in case of static nodes
        if isinstance(node, Static):
            raise ValueError('Static nodes do not have any input')

        input_bindings = Graph.expand(input_bindings)   # TODO run adapter here
        for n in multi_iterable_get_values_as_list(input_bindings):
            graph.add_node_if_obj(n)

        input_bindings = multi_iterable_map(Node.get_name, input_bindings)
        graph.link(node, input_bindings)

    if isinstance(node, Node):
        return node


class Lambda(Node):
    def __init__(self, name=None, func=None):
        if not callable(func):
            raise ValueError("Param func should be a callable")
        self.func = func
        super(Lambda, self).__init__(name)

    def get_hpopt_config_ranges(self):
        return {}

    def __call__(self, input, hpopt_config={}):
        return self.func(input)


def lambda_(f) -> Node:
    """
    Creates a lambda node. Note it is also useful to create callbacks by behaving like an identity.
    :param f: The callable invoked by the node. The only input argument is the input of the node, the output of the
    callable is returned as node's output.
    :return: The lambda node
    """
    node = Lambda(func=f)
    Graph.get_default().add_node(node)
    return node


def multi_iterable_map(fn, iterable):
    if isinstance(iterable, list):
        return [fn(obj) for obj in iterable]
    if isinstance(iterable, dict):
        return dict([(key, fn(obj)) for key, obj in iterable.items()])
    return fn(iterable)


def multi_iterable_get_values_as_list(iterable):
    if isinstance(iterable, list):
        return iterable
    if isinstance(iterable, dict):
        return list(iterable.values())
    return [iterable]


def multi_iterable_get_keys_as_list(iterable):
    if isinstance(iterable, list):
        return iterable
    if isinstance(iterable, dict):
        return list(iterable.keys())
    return [iterable]


class Variable(Node):
    def __init__(self, name, dtypes=None):       # TODO initial value!
        self._dtypes = dtypes
        super(Variable, self).__init__(name)

    def get_hpopt_config_ranges(self):
        return {}

    @property
    def dtypes(self):
        return self._dtypes

    def set_value(self, ctx=None, value=None):
        ctx = ExecutionContext.get_instance(ctx)
        ctx.set_var(graph=self.parent, var=self, value=value)

    def check_type(self, value):
        if self.dtypes is not None:
            if not isinstance(value, self.dtypes):
                raise ValueError('The variable has a restricted set of allowed types')

    def __call__(self, input, hpopt_config={}):
        return ExecutionContext.get_default().get_var_value(self)


#TODO class Subgraph(Node)


class GraphCallback:
    def on_node_execution_end(self, ctx: ExecutionContext, graph, **args):
        pass

    def on_node_add(self, graph, node: Node, **args):
        pass


class Graph(Node):
    _default = None

    #TODO serializer & deserializer

    @classmethod
    def get_default(cls):
        """
        Returns the default graph. If there is no default graph then a new one is instantiated
        :return:
        """
        if cls._default is None:
            cls._default = Graph()
        return cls._default

    @classmethod
    def reset_default(cls):
        """
        Resets the default graph
        :return:
        """
        cls._default = None

    def __init__(self, id=None, default_output=None, callback: GraphCallback=None):
        self.nodes = {}
        self.lazy_links = {}    # key is the node name, value is an input binding
        self.default_output = default_output
        self.callback = callback
        super(Graph, self).__init__(id)

    @contextmanager
    def as_default(self):
        """
        Returns a context manager that sets this graph as default
        :return: A context manager
        """
        prev_default = Graph._default
        try:
            Graph._default = self
            yield
        finally:
            Graph._default = prev_default

    def link(self, node, input_binding=None):
        """
        If node is an identifier then a lazy binding is created. In the case the node is an instance of Node
        then the set_input_binding is invoked directly on the destination node
        :param node: A node identifier or a Node object
        :param input_binding:
        :return: If the node is an instance of Node then the object is returned otherwise None
        """
        if isinstance(node, Node):
            # In the case the node object is provided directly we remove any lazy link and we set the
            # input bindings directly
            self.lazy_links.pop(node.namespace, None)
            node.set_input_binding(input_binding)
            return

        if not isinstance(node, str):
            raise ValueError('The node name is not a string')
        self.lazy_links[node] = input_binding

    # TODO remove input binding from Nodes? How to deal with hpopt?
    # TODO create another runtime link in the execution context?
    #def get_input_binding_for_node(self, node):
    #    name = Node.get_name(node)
    #    return copy.copy(self.lazy_links.get(name))

    def add_node_if_obj(self, node):
        if isinstance(node, str):
            return
        self.add_node(node)

    def add_node(self, node, input_binding=None):
        """
        Adds a node and the relative input bindings to the graph
        :param node: An object of type Node
        :param input_binding: A string, a list of strings or a dictionary representing the binding of the input of this node
        with respect to the output of another node(s)
        :return:
        """

        # TODO adapter, if node is not an instance of Node, look for an adapter..., call adapter in the expand function

        if node is self:
            raise ValueError('A graph cannot be a child of itself')

        if isinstance(node, Node):
            if node._parent is not None and node._parent!=self:
                raise ValueError('The node is already part of another Graph')
            if input_binding is not None:
                node.set_input_binding(input_binding)

            #TODO check if there is a name conflict
            self.nodes[node.namespace] = node
            node._parent = self

            if self.callback is not None:
                self.callback.on_node_add(graph=self, node=node)
            return

        raise ValueError('The node argument must be an instance of the class Node')

    def get_hpopt_config_ranges(self):
        output = {}
        for key, node in self.nodes.items():
            output.update(node.get_hpopt_config_ranges())
        return output

    @staticmethod
    def get_node_ext(node, graph=None):
        if graph is not None:
            if not isinstance(graph, Graph):
                raise ValueError()
            return graph.get_node(node)
        if isinstance(node, Node):
            return node
        raise ValueError('Not enough data provided')

    def get_node(self, node):
        """
        If node is an identifier of a node it looks up for it and returns the associated node. If node is an instance
        of Node then ownership by this graph is checked and the node is returned
        :param node:
        :return:
        """
        ownership_err_str = 'This node is not owned by this graph'

        if isinstance(node, Node):
            if not isinstance(node, Graph) and (node.parent is not self):
                raise ValueError(ownership_err_str)
            return node

        if not isinstance(node, str):
            raise ValueError('Param node must be a string identifier or an object of the instance Node')

        node = self.nodes.get(node)
        if node is None:
            raise ValueError(ownership_err_str)
        return node

    def get_var(self, var):
        var = self.get_node(var)
        if not isinstance(var, Variable):
            raise ValueError()
        return var

    class _Executor:
        def __init__(self, parent, graph_input):
            assert isinstance(parent, Graph)
            self.parent = parent
            self.graph_input = graph_input
            self.node_output_map = {}

        def lookup_output(self, node):
            """
            Returns the output of the execution of a specific node. This is a helper function to be used by multi_iterable_map
            :param node: An identifier of a node or a Node object
            :return: The node output
            """
            name = Node.get_name(node)
            return self.node_output_map.get(name)

        def _callback_on_node_exec(self, node):
            parent = self.parent
            if parent.callback is not None:
                parent.callback.on_node_execution_end(ExecutionContext.get_default(), parent,
                                                      {'node_output_map': self.node_output_map,
                                                       'node': node})

        def clear_non_pure_outputs(self):
            filtered_nodes = filter(lambda n: not n.flags.contains('p'),
                                    [self.parent.nodes[k] for k in self.node_output_map.keys()])
            for n in filtered_nodes:
                self.node_output_map.pop(n.namespace)

        def run_node(self, node, hpopt_config={}):
            """
            Executes one node (after the execution of its requirements) and store the output in node_output_map
            :param node: Can be an identifier or an object of class Node
            :param hpopt_config: the hyper-parameters selected by hyperopt
            :return:
            """

            parent = self.parent
            node = parent.get_node(node)
            if node.namespace in self.node_output_map:
                # this node has already been processed
                return

            if isinstance(node, Placeholder):
                self.node_output_map[node.namespace] = node.select_input(self.graph_input)
                self._callback_on_node_exec(node)
                return

            # now lookup the input binding
            input_binding = parent.lazy_links.get(node.namespace)  # the lazy binding has the priority
            if input_binding is None:
                input_binding = node.get_input_binding(hpopt_config)

            if input_binding is None:
                # the node doesn't have an input
                self.node_output_map[node.namespace] = node(None, hpopt_config)
                return

            requirements = multi_iterable_get_values_as_list(input_binding)
            for req in requirements:
                req_name = Node.get_name(req)
                if req_name not in parent.nodes:
                    raise ValueError('A node mentioned in the input bindings is not present in the current graph, '
                                     'name=\'{}\''.format(req_name))

                if req_name not in self.node_output_map:
                    #TODO remove recursion
                    self.run_node(req_name, hpopt_config)
                    assert req_name in self.node_output_map

            # All requirements are fulfilled so exec the node and store the output
            self.node_output_map[node.namespace] = node(multi_iterable_map(self.lookup_output, input_binding),
                                                        hpopt_config=hpopt_config)
            self._callback_on_node_exec(node)

        def run(self, nodes, hpopt_config={}):
            for req in multi_iterable_get_values_as_list(nodes):
                self.run_node(req, hpopt_config)
            return multi_iterable_map(self.lookup_output, nodes)

    def __call__(self, input=None, hpopt_config={}, outputs=None):
        """
        Runs the output nodes and their dependencies
        :param input:
        :param outputs: A list/dict or string of identifiers of the required outputs, if None the default list of
        outputs is used
        :param hpopt_config:
        :return:
        """
        outputs = self.default_output if outputs is None else outputs
        if outputs is None:
            return None

        return self._Executor(self, input).run(outputs, hpopt_config)

    @classmethod
    def expand(cls, obj):
        vm = json_vm.VM('#$#', cls.operators_reg)
        obj = vm.run(obj)
        adapter = cls.adapters.get(type(obj))   # TODO check key using issubclass()
        if adapter:
            return adapter(obj)
        # TODO if no adapter present and not Node or string, encapsulate into a static automatically???
        return obj


def return_(collection):
    """
    Sets the default output of the current graph
    :param collection: A node or an identifier or a list or dictionary of the same
    :return:
    """
    Graph.get_default().default_output = collection


Graph.operators_reg = {
    '#$#g.dummy': Dummy.deserializer,
    '#$#g.placeholder': Placeholder.deserializer,
    '#$#g.placeholder_all': partial(Placeholder.deserializer, match_all_inputs=True),
    '#$#g.identity': Identity.deserializer,
    '#$#g.switch': Switch.deserializer,
    '#$#g.static': Static.deserializer,
    '#$#g.return': return_,
    '#$#g.get_keys': get_keys,
    '#$#g.get_input_keys': get_input_keys,
    '#$#g.merge': merge
}

Graph.adapters = {  # A map type: adapter
    types.FunctionType: lambda_,
    # TODO Graph: Subgraph
}
