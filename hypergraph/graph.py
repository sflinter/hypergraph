from . import json_vm
from abc import ABC, abstractmethod
import itertools
from contextlib import contextmanager
from functools import reduce
from functools import partial as fpartial
import types
import weakref
from .utils import export
import inspect
import uuid


# Separator used in fully qualified names
FQ_NAME_SEP = '.'


def fq_ident(idents, sep=FQ_NAME_SEP) -> str:
    """
    Create a fully qualified identifier
    :param idents: list of identifiers or a string
    :param sep: the separator for the identifiers
    :return: the fully qualified identifier as string
    """
    if isinstance(idents, str):
        return idents
    return reduce(lambda s1, s2: s1 + sep + str(s2), idents)


@export
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

    # TODO node context...
    def __init__(self, tweaks={}):
        if not isinstance(tweaks, dict):
            raise ValueError()
        self._vars_per_graph = {}
        self.tweaks = tweaks

    def set_var(self, graph=None, var=None, value=None):
        var = Graph.get_node_ext(var, graph)
        graph = var.parent
        if not isinstance(var, Variable):
            raise ValueError('The node var must be of type Variable')

        vars_ = self._vars_per_graph.setdefault(graph, {})
        var.check_type(value)
        vars_[var.name] = value

    def get_var_value(self, graph=None, var=None):
        var = Graph.get_node_ext(var, graph)
        graph = var.parent
        if not isinstance(var, Variable):
            raise ValueError('The node var must be of type Variable')

        vars_ = self._vars_per_graph.get(graph)
        if vars_ is None:
            return None
        return vars_[var.name]


@export
class Node(ABC):
    _id_counter = itertools.count()

    def __init__(self, name=None, flags=()):
        if name is None:
            name = 'node-'+str(next(self._id_counter))
        if not isinstance(name, str):
            raise ValueError('Node name must be a string')

        self._name = name
        self._parent_wref = None     # the graph that contains this node
        self._flags = flags

        if self is not Graph.get_default():
            Graph.get_default().add_node(self)

    @staticmethod
    def get_name(node) -> str:
        if isinstance(node, Node) or isinstance(node, NodeId):
            return node.name
        raise ValueError('The input param must be either a NodeId or a Node, '
                         'the current type is {}'.format(type(node)))

    @staticmethod
    def nodeId(node):
        if isinstance(node, NodeId):
            return node
        if isinstance(node, Node):
            return NodeId(node.name)   # TODO use fully qualified name!
        raise ValueError('The input param must be either a NodeId or a Node, '
                         'the current type is {}'.format(type(node)))

    @property
    def name(self) -> str:
        return self._name

    @property
    def fully_qualified_name(self) -> str:
        g = self.parent
        if g is None:
            raise RuntimeError()
        return g.name + FQ_NAME_SEP + self.name

    @property
    def consumed(self):
        """
        Indicates whether this node is part of a graph
        :return:
        """
        return self._parent_wref is not None

    @property
    def parent(self):
        """
        This property returns the graph that contains this node
        :return: a Graph
        """
        ref = self._parent_wref
        if ref is not None:
            return ref()
        return None

    @property
    def flags(self) -> tuple:
        return self._flags

    def get_input_binding(self, hpopt_config={}):
        g = self.parent
        assert g is not None
        return g.get_node_input_binding(self)

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
        This method executed the node, note that the internal permanent status should be stored in the current
        execution context.
        :param input: The input(s) to this node
        :param hpopt_config: A dictionary with the values choosed by the hyper parameters optimizer, the params that aren't
        specified will take the default value (from definition)
        :return:
        """
        pass

    def __getitem__(self, key):
        """
        Return a get_key node linked to the current one with the specified key
        :param key:
        :return:
        """
        return get_key(key, _input=self)

    def __lshift__(self, other): # TODO do the same with NodeId!!!
        if isinstance(other, Dependency):
            other.inject(self)
            return self
        return link(self, other)

    def serialize(self):
        """
        Serializes the node into a json serializable format
        :return:
        """
        raise NotImplementedError()


class NonExecutable(Node):
    def __call__(self, input, hpopt_config={}):
        raise RuntimeError("This node is not supposed to be executed directly")


class Identity(Node):
    """
    A simple node that returns the input when invoked without any modification.
    """

    def __init__(self, name=None):
        super(Identity, self).__init__(name)

    @staticmethod
    def deserializer(id=None):
        return Identity(id)

    def __call__(self, input, hpopt_config={}):
        return input


@export
def mark(name=None) -> Node:
    """
    Mark is a pseudonym of identity. The idea is to mark internal placeholders (from the graph point of view)
    :param name: The identifier of the identity node
    :return:
    """

    return Identity(name)


class InputPlaceholder(NonExecutable):
    """
    A node that represents the input of a graph
    """
    def __init__(self, key=None, match_all_inputs=False):
        self.key = key
        self.match_all_inputs = bool(match_all_inputs)
        super(InputPlaceholder, self).__init__()

    def select_input(self, input):
        if self.match_all_inputs:
            return input
        return input[self.key]

    @staticmethod
    def deserializer(key=None, match_all_inputs=False):
        return InputPlaceholder(key, match_all_inputs)


@export
def input_key(key=None):
    # TODO provide also: input()[key]?
    return InputPlaceholder(key, match_all_inputs=False)


@export
def input_all():
    #TODO generate just one node placeholder_all per graph
    return InputPlaceholder(key=None, match_all_inputs=True)


class Merge(Node):
    def __init__(self, name=None, mode='d'):
        """
        Init merge node.
        :param name:
        :param mode: 'd': dictionary mode
        """
        if not(mode in ('d', )):
            raise ValueError()
        self.mode = mode
        super(Merge, self).__init__(name)

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

        raise ValueError()


@export
def merge(mode):
    return Merge(name=None, mode=mode)


@export
class NodeId:
    # TODO include "graph id" (to be decided) although the id can be either fully qualified and relative
    def __init__(self, name):
        if not isinstance(name, str):
            raise ValueError()
        self._name = name

    @property
    def name(self) -> str:
        return self._name


@export
def node(node):
    node = Graph.adapt(Graph.expand(node))

    if isinstance(node, Node):
        return node
    if isinstance(node, (NodeId, str)):
        return Graph.get_default().get_node(node)

    # adapt everything else
    return link(Identity(), node)


@export
def node_ref(node):
    if isinstance(node, str):
        return NodeId(node)
    if isinstance(node, NodeId):
        return node
    if isinstance(node, Node):
        # Graph.get_default().add_node(node)
        return NodeId(node.name)

    # adapt everything else
    node = link(Identity(), node)
    return NodeId(node.name)


class GetKeys(Node):
    """
    Gets elements from lists or dictionaries
    """

    def __init__(self, name=None, keys=None, output_type=None):
        if not(output_type in [None, 'd']):
            raise ValueError()
        self.keys = keys
        self.output_type = output_type
        super(GetKeys, self).__init__(name)

    @staticmethod
    def deserializer(keys, output_type=None):
        return GetKeys(name=None, keys=keys, output_type=output_type)

    def __call__(self, input, hpopt_config={}):
        if not isinstance(self.keys, list):
            return input[self.keys]

        if self.output_type == 'd':
            values = [input[k] for k in self.keys]
            return dict(zip(self.keys, values))

        return [input[k] for k in self.keys]


class _RuntimeKey:
    """
    Internal placeholder class used to indicate that a key is a graph runtime value
    """
    pass


class GetKey(Node):
    """
    Get an element from a subscriptable
    """

    def __init__(self, name=None, key=_RuntimeKey):
        self.key = key
        super().__init__(name)

    @property
    def is_key_runtime(self):
        return self.key == _RuntimeKey

    def __call__(self, input, hpopt_config={}):
        if self.key is _RuntimeKey:
            subscriptable = input['subscriptable']
            key = input['key']
            return subscriptable[key]
        else:
            return input[self.key]


@export
def get_keys(keys, output_type=None):
    return GetKeys(name=None, keys=keys, output_type=output_type)


def get_key(key, _input=None):
    if len(get_nodes_from_struct(key)) == 0:
        # optimization for static types
        if _input is not None:
            return GetKey(key=key) << _input
        return GetKey(key=key)

    # param _input must be provided
    if _input is Node:
        raise ValueError("Invalid combination of parameters")

    # TODO put key into the deps so it is evaluated first and we avoid the eval of the entire
    # subscriptable?
    return link(GetKey(), {'subscriptable': _input, 'key': key})

@export
def input_keys(keys, output_type=None):
    # TODO provide also: get_keys(input(), ...)
    return link(get_keys(keys=keys, output_type=output_type), input_all())


class Dump(Node):
    def __init__(self, name=None):
        super().__init__(name)

    @staticmethod
    def deserializer(name=None):
        return Dump(name)

    def __call__(self, input, hpopt_config={}):
        """
        Dump the input and return it so it can be used transparently.
        :param input:
        :param hpopt_config:
        :return:
        """
        print(input)
        return input


@export
def dump(name=None):
    # TODO optional static message to display?
    return Dump(name=name)


@export
def link(node, input_bindings=None, deps=None) -> [Node, None]:
    """
    Creates a (lazy) link in the current graph between the node and the nodes identified by the input_bindings
    :param node: A string identifier of the node or a Node object
    :param input_bindings:
    :param deps:
    :return: node if node is an instance of Node
    """

    graph = Graph.get_default()

    node = Graph.expand(node)
    graph.add_node_if_obj(node)
    # TODO remove add, check ownership instead

    if input_bindings is not None:
        input_bindings = Graph.expand(input_bindings)
        for n in get_nodes_from_struct(input_bindings):
            graph.add_node_if_obj(n)

        input_bindings = substitute_nodes(input_bindings, Node.nodeId)
    graph.link(node, input_bindings)

    graph.clear_dep(node)
    if deps is not None:
        deps = get_nodes_from_struct(Graph.expand(deps))
        for n in deps:
            graph.add_node_if_obj(n)
        graph.add_dep(node, substitute_nodes(deps, Node.nodeId))

    if isinstance(node, Node):
        return node
    return Graph.nodeId(node)


@export
def add_event_handler(event, deps):
    if deps is None:
        raise ValueError()

    graph = Graph.get_default()

    deps = get_nodes_from_struct(Graph.expand(deps))
    for n in deps:
        graph.add_node_if_obj(n)

    graph.add_event_handler(event, substitute_nodes(deps, Node.nodeId))


class Lambda(Node):
    """
    A node that encapsulates a lambda function. Another possibility is using the node function: node(lambda x: ...)
    The only input argument of the lambda is the input of the node, the output of the callable is returned as node's output.
    """

    def _inspect_func(self):
        sig = inspect.signature(self.func)
        self.params = sig.parameters.keys()

    def __init__(self, name=None, func=None, map_arguments=False):
        if not callable(func):
            raise ValueError("Param func should be a callable")
        self.func = func
        self.map_arguments = map_arguments  # or hasattr(func, '_hg_func_node_tag')
        self.params = None

        self._inspect_func()
        super().__init__(name)

    def __call__(self, input, hpopt_config={}):
        if self.map_arguments is False:
            return self.func(input)

        if isinstance(input, dict):
            # TODO use self.params, if some flag is set...
            return self.func(**input)

        return self.func(*input)


@export
def call(func, name=None):
    return Lambda(func=func, name=name, map_arguments=True)


@export
def call1(func, name=None):
    return Lambda(func=func, name=name, map_arguments=False)


def multi_iterable_map(fn, iterable):
    if isinstance(iterable, (list, tuple)):
        return [fn(obj) for obj in iterable]
    if isinstance(iterable, dict):
        return dict([(key, fn(obj)) for key, obj in iterable.items()])
    return fn(iterable)


def substitute_nodes(nodes, lookup_fn):
    if isinstance(nodes, (list, tuple)):
        output = []
        for obj in nodes:
            if isinstance(obj, (NodeId, Node)):
                output.append(lookup_fn(obj))
            elif isinstance(obj, (list, dict, tuple)):
                output.append(substitute_nodes(obj, lookup_fn))
            else:
                output.append(obj)
        return output
    elif isinstance(nodes, dict):
        output = {}
        for key, obj in nodes.items():
            if isinstance(obj, (NodeId, Node)):
                output[key] = lookup_fn(obj)
            elif isinstance(obj, (list, dict, tuple)):
                output[key] = substitute_nodes(obj, lookup_fn)
            else:
                output[key] = obj
        return output
    elif isinstance(nodes, (NodeId, Node)):
        return lookup_fn(nodes)
    else:
        return nodes


def get_nodes_from_struct(iterable, output=None):
    if output is None:
        output = []

    def op(obj):
        if isinstance(obj, (Node, NodeId)):
            output.append(obj)
        elif isinstance(obj, (list, tuple)):
            get_nodes_from_struct(obj, output)
        elif isinstance(obj, dict):
            get_nodes_from_struct(obj, output)

    if isinstance(iterable, (list, tuple)):
        for obj in iterable:
            op(obj)
    elif isinstance(iterable, dict):
        for obj in iterable.values():
            op(obj)
    elif isinstance(iterable, (Node, NodeId)):
        output.append(iterable)

    return output


def struct_copy(iterable):
    # TODO remove recursion, use deepcopy strategy
    if isinstance(iterable, (list, tuple)):
        # Note, tuple is converted to list
        return [struct_copy(item) for item in iterable]
    elif isinstance(iterable, dict):
        return dict([(k, struct_copy(v)) for k, v in iterable.items()])
    else:
        return iterable


@export
class Variable(Node):
    def __init__(self, name, dtypes=None):       # TODO initial value!
        self._dtypes = dtypes
        super().__init__(name)

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


class Jump(NonExecutable):
    def __init__(self, name=None, destination=None):
        """
        Create a Jump node
        :param name:
        :param destination: node id of the static destination node, if None, then the destionation node is dynamic
        """
        self.destination = None
        if destination is not None:
            self.destination = node_ref(destination)
        super().__init__(name)


@export
def jmp(destination=None) -> Jump:
    return Jump(destination=destination)


class Indirect(NonExecutable):
    def __init__(self, name=None, input_node=None, output_node=None):
        self.input_node = node_ref(input_node)
        self.output_node = node_ref(output_node)
        super().__init__(name)


@export
def indirect(input_node=None, output_node=None) -> Indirect:
    return Indirect(input_node=input_node, output_node=output_node)


class Partial(NonExecutable):
    def __init__(self, partial_input, name=None):
        self.partial_input = partial_input
        super().__init__(name)


@export
def partial(partial_input):
    return Partial(partial_input=partial_input)


@export
def select(idx_node, nodes=None) -> Node:
    """
    Select a node from a list or dict (provided as input) and jump the execution to it. If param nodes is specified
    then the returned node is a head, thus do not provide an input.
    :param idx_node: A node that returns the index
    :param nodes: A list of nodes identifiers
    :return:
    """
    if nodes is not None:
        nodes = mark() << nodes
        return jmp() << nodes[idx_node]

    input = mark()
    output = jmp() << link(GetKey(), {'subscriptable': input, 'key': idx_node})
    output = mark() << output   # indirect return to avoid Graph._Executor.run_node double alias
    return indirect(input_node=input, output_node=output)


def _first_valid(iterable):
    if isinstance(iterable, dict):
        for item in iterable.items():
            if item[1]:
                return item[0]
    for item in iterable:
        if item:
            return item


@export
def first_valid():
    return Lambda(func=_first_valid)


class GraphCallback:
    def on_node_execution_end(self, ctx: ExecutionContext, graph, **args):
        pass

    def on_node_add(self, graph, node: Node, **args):
        pass


@export
@contextmanager
def sequential_link():
    g = Graph.get_default()
    prev_value = g.sequential_link_prev_node
    g.sequential_link_prev_node = 0     # when this var is not None then sequential linking is enabled
    yield
    g.sequential_link_prev_node = prev_value


class Subgraph(Node):
    def __init__(self, name=None, graph=None):
        if not isinstance(graph, Graph):
            raise ValueError()
        self.graph = graph
        super().__init__(name)

    def get_hpopt_config_ranges(self):
        return self.graph.get_hpopt_config_ranges()

    def __call__(self, input, hpopt_config={}):
        return self.graph(input)


class Dependency:
    """
    A class used to inject node dependencies using the shift operator
    """
    def __init__(self, deps):
        self.deps = deps

    def inject(self, node):
        graph = Graph.get_default()

        deps = self.deps
        deps = get_nodes_from_struct(Graph.expand(deps))
        for n in deps:
            graph.add_node_if_obj(n)
        graph.add_dep(node, substitute_nodes(deps, Node.nodeId))


@export
def deps(deps) -> Dependency:
    """
    Inject dependencies using the node's shift operator, example: node << deps([node_ref('abc'), cde])
    :param deps:
    :return:
    """
    return Dependency(deps)


@export
class Graph:
    _default = None

    @classmethod
    def get_default(cls, name=None):
        """
        Returns the default graph. If there is no default graph then a new one is instantiated
        :return:
        """
        if cls._default is None:
            cls._default = Graph(name=name)
        return cls._default

    @classmethod
    def reset_default(cls):
        """
        Resets the default graph
        :return:
        """
        cls._default = None

    def __init__(self, name=None, default_output=None, callback: GraphCallback=None):
        """
        Init a new graph.
        :param name: The name of the graph, this will be used to form the fully qualified names for the nodes.
        :param default_output:
        :param callback:
        """

        # TODO specifiy parallel=True|False, this enable/disable the parallel execution of graph nodes.
        # By parallel execution we mean a generic one (thus a handler should be passed) and not the classical
        # multi-thread execution.

        self.nodes = {}    # key is the node name, value is a node object
        self.links = {}    # key is the node name, value is an input binding
        self.deps = {}     # key is the node name, value is a list of node names
        self.event_handlers = {}    # key is event name, value is a node object
        self.default_output = default_output    # TODO rename output, this would override the internal graph's output
        self.callback = callback
        self.sequential_link_prev_node = None

        if name is None:
            name = 'hggraph-'+str(uuid.uuid4())
        if not isinstance(name, str):
            raise ValueError("Graph name is expected to be a string")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

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

    def add_dep(self, node, dependencies):
        """
        Add a list of dependencies to a specific node. Dependencies are executed before the current node.
        Input bindings are themselves dependencies but are considered separately.
        :param node: The dependent node
        :param dependencies: A node or a list of nodes
        :return:
        """
        node = Node.get_name(node)
        if isinstance(dependencies, list):
            self.deps.setdefault(node, set()).update(map(Node.get_name, dependencies))
        else:
            self.deps.setdefault(node, set()).add(Node.get_name(dependencies))

    def remove_dep(self, node, dependencies):
        # TODO
        raise NotImplementedError()

    def clear_dep(self, node):
        """
        Remove all dependencies for the specified node
        :param node:
        :return:
        """
        self.deps.setdefault(node, set()).clear()

    def get_node_deps(self, node):
        """
        Return a list of dependencies relative to the current node.
        :param node: The dependent node
        :return:
        """
        return list(self.deps.get(Node.get_name(node), set()))

    def link(self, node, input_binding=None):
        """
        If node is an identifier then a lazy binding is created. In the case the node is an instance of Node
        then the set_input_binding is invoked directly on the destination node
        :param node: A node identifier or a Node object
        :param input_binding:
        :return: If the node is an instance of Node then the object is returned otherwise None
        """

        # TODO check whether node is instance of InputPlaceholder
        if isinstance(node, Indirect):
            node = node.input_node

        node = Node.get_name(node)
        assert isinstance(node, str)
        self.links[node] = struct_copy(input_binding)

    def add_event_handler(self, event, nodes):
        if event not in self.event_types:
            raise ValueError()
        handlers = self.event_handlers.setdefault(event, set())
        for node in substitute_nodes(get_nodes_from_struct(nodes), self.get_node):
            handlers.add(node)

    def remove_event_handler(self, event, nodes):
        handlers = self.event_handlers.get(event)
        for node in substitute_nodes(get_nodes_from_struct(nodes), self.get_node):
            handlers.discard(node)

    def add_node_if_obj(self, node):
        if isinstance(node, NodeId):
            return
        self.add_node(node)

    def _auto_link(self, node):
        """
        Il sequential linking is enabled, then link automatically current node to the previous one.
        :param node: current node
        :return:
        """
        prev_node = self.sequential_link_prev_node
        if prev_node is not None:
            if isinstance(prev_node, Node):
                self.link(node, prev_node)
            self.sequential_link_prev_node = node

    def add_node(self, node):
        """
        Adds a node and the relative input bindings to the graph
        :param node: An object of type Node
        :param input_binding: A string, a list of strings or a dictionary representing the binding of the input of this node
        with respect to the output of another node(s)
        :return:
        """

        # TODO allow a subgraph(self) to be a node of itself?

        if isinstance(node, Node):
            if node.parent is not None:
                if node.parent != self:
                    raise ValueError('The node is already part of another Graph')
                else:
                    self._auto_link(node)
                    return

            if node.name in self.nodes:
                raise ValueError('Duplicate node name error')
            self.nodes[node.name] = node
            node._parent_wref = weakref.ref(self)

            if self.callback is not None:
                self.callback.on_node_add(graph=self, node=node)
            self._auto_link(node)
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

    def get_node_input_binding(self, node):
        node = Node.get_name(node)
        input_binding = self.links.get(node)
        if input_binding is not None:
            input_binding = struct_copy(input_binding)
        return input_binding

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

        if isinstance(node, NodeId):
            node = node.name
        # Note: get_node(str) is a valid possibility
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
            self.ctx = ExecutionContext.get_default()

        def lookup_output(self, node):
            """
            Returns the output of the execution of a specific node. This is a helper function to be used by substitute_nodes
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
                self.node_output_map.pop(n.name)

        def _solve_requirements(self, input_binding):
            parent = self.parent

            requirements = get_nodes_from_struct(input_binding)
            for req in requirements:
                req_name = Node.get_name(req)
                if req_name not in parent.nodes:
                    raise ValueError('A node mentioned in the input bindings is not present in the current graph, '
                                     'name=\'{}\''.format(req_name))

                if req_name not in self.node_output_map:
                    #TODO remove recursion
                    self.run_node(req_name)
                    assert req_name in self.node_output_map

        def _substitution(self, input_binding):
            return substitute_nodes(input_binding, lookup_fn=self.lookup_output)

        def _set_node_output(self, node, value, alias=None):
            # TODO remove alias, pass a list of nodes instead?
            if node is not None:
                self.node_output_map[node.name] = value
                self._callback_on_node_exec(node)
            if alias is not None:
                self.node_output_map[Node.get_name(alias)] = value
                self._callback_on_node_exec(self.parent.get_node(alias))

        def _handle_partial(self, node: Partial):
            partial_input = node.partial_input
            self._solve_requirements(partial_input)
            input_binding = node.get_input_binding(self.ctx.tweaks)
            self._solve_requirements(input_binding)
            output = [self._substitution(partial_input), self._substitution(self._substitution(input_binding))]
            self._set_node_output(node, value=output)

        def _handle_jmp(self, node: Jump):
            input_binding = node.get_input_binding(self.ctx.tweaks)
            if node.destination is not None:
                if input_binding is not None:
                    raise RuntimeError("Jump nodes are not supposed to receive an input when"
                                       " a static target is specified")
                self.run_node(node.destination, alias=node)
                return

            self._solve_requirements(input_binding)
            destination = self._substitution(input_binding)
            destination = node_ref(destination)
            destination = self.parent.get_node(destination)
            self.run_node(destination, alias=node)

        def _handle_indirect(self, node: Indirect):
            self.run_node(node.output_node, alias=node)

        def _run_deps(self, node):
            for d in self.parent.get_node_deps(node):
                # TODO in the future these will be executed in parallel
                self.run_node(d)

        def run_node(self, node, alias=None):
            """
            Executes one node (after the execution of its requirements) and store the output in node_output_map
            :param node: Can be an identifier or an object of class Node
            :param alias: A node (or node id) that receives the same output of the current node
            :return:
            """

            node = self.parent.get_node(node)
            self._run_deps(node)

            if node.name in self.node_output_map:
                # this node has already been processed
                if alias is not None:
                    self._set_node_output(node=None, value=self.node_output_map[node.name], alias=alias)
                return

            # *** begin of custom nodes handling ***
            if isinstance(node, InputPlaceholder):
                self._set_node_output(node, value=node.select_input(self.graph_input), alias=alias)
                return

            if isinstance(node, Jump):
                assert alias is None    # TODO what's happen if we have jmp after a jmp?!?!?!
                self._handle_jmp(node)
                return

            if isinstance(node, Partial):
                assert alias is None    # TODO what's happen if alias!=None?
                self._handle_partial(node)
                return

            if isinstance(node, Indirect):
                assert alias is None    # TODO what's happen if alias!=None?
                self._handle_indirect(node)
                return
            # *** end of custom nodes handling ***

            input_binding = node.get_input_binding(self.ctx.tweaks)
            if input_binding is None:
                # the node doesn't have an input
                self._set_node_output(node, value=node(None, hpopt_config=self.ctx.tweaks), alias=alias)
                return

            self._solve_requirements(input_binding)

            # All requirements are fulfilled so exec the node and store the output
            self._set_node_output(node, value=node(self._substitution(input_binding), hpopt_config=self.ctx.tweaks),
                                  alias=alias)

        def run_event_handlers(self, event):
            handlers = self.parent.event_handlers.get(event)
            if handlers is not None:
                for h in handlers:
                    self.run_node(h)

        def run(self, nodes):
            # Execute on enter handlers
            self.run_event_handlers('enter')

            for req in get_nodes_from_struct(nodes):
                self.run_node(req)
            output = substitute_nodes(nodes, self.lookup_output)

            # Execute on exit handlers
            self.run_event_handlers('exit')

            return output

    def __call__(self, input=None, *, outputs=None):
        """
        Runs the output nodes and their dependencies
        :param input:
        :param outputs: A list/dict or string of identifiers of the required outputs, if None the default list of
        outputs is used
        :return:
        """
        outputs = self.default_output if outputs is None else outputs
        if outputs is None:
            return None

        # TODO call user specified context manager here (useful for log)
        return self._Executor(self, input).run(outputs)

    @classmethod
    def expand(cls, obj):
        vm = json_vm.VM('#$#', cls.operators_reg)
        return vm.run(obj)

    @classmethod
    def adapt(cls, obj):
        adapter = cls.adapters.get(type(obj))   # TODO check key using issubclass()
        if adapter:
            return adapter(obj)
        return obj


@export
def output(collection=None, deps=None) -> Node:
    """
    Set the default output of the current graph.
    :param collection: A node or an identifier or a list or dictionary of the same, this represents the output,
    if collection is None instead, then a special node representing the output is returned.
    :return:
    """
    if collection is None:
        if deps is not None:
            raise ValueError("Invalid combination of parameters")
        n = Identity()
        Graph.get_default().default_output = n
        return n

    output = link(Identity(), collection, deps=deps)
    Graph.get_default().default_output = output


Graph.operators_reg = {
    '#$#g.dump': Dump.deserializer,
    '#$#g.input': fpartial(InputPlaceholder.deserializer, match_all_inputs=True),
    '#$#g.identity': Identity.deserializer,
    '#$#g.output': output,
    '#$#g.get_keys': get_keys,
    '#$#g.input_keys': input_keys,
    '#$#g.merge': merge
}

Graph.event_types = {'enter', 'exit'}   # TODO exception handler

Graph.adapters = {  # A map type: adapter
    types.FunctionType: lambda f: Lambda(func=f),
    Graph: lambda g: Subgraph(graph=g)
}
