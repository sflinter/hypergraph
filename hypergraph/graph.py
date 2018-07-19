from . import json_vm
from abc import ABC, abstractmethod
import itertools
from contextlib import contextmanager
from functools import partial, reduce
import types
import weakref
import copy

# TODO create Pretrained weights node, think about the dependency from Resnet type...
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

# TODO parallel execution (even on multiple servers) of dependencies and input bindings!!!


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


class Node(ABC):
    _id_counter = itertools.count()

    def __init__(self, name=None, flags=()):   # TODO rename name?
        if name is None:
            name = 'node-'+str(next(self._id_counter))
        if not isinstance(name, str):
            raise ValueError('Node name must be a string')

        self._ns = name
        self._parent_wref = None     # the graph that contains this node
        self._flags = flags

        #if self is not Graph.get_default():
        #    Graph.get_default().add_node(self)

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
        return self._ns

    # TODO fully_qualified_name

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

    # TODO remove hpopt_config_ranges, create a more generic idea,
    # for example provide node's "tweaks" (switches) relative to a specific context
    # one of the contexes is hpopt. Note that input bindings may depends on a particular "tweak"
    # so, first we apply tweaks and then we build the final links. The tweaks may override
    # the links declared in the graph, the custom links are declared in the execution context.
    # When we return a tweak configuration we return a list or dictionary (or a more complex structure)
    # where the user value are substituted by placeholders. Placeholders like hyperopt may contain a range of validity

    def get_hpopt_config_ranges(self) -> dict:
        """
        Return a dictionary with the hyperopt ranges. The keys must be fully qualified identifiers of the internal
        options. This should not be invoked for a non-consumed node, that is a node not added to a graph (this because
        some ranges may depend from input bindings).
        :return: A dictionary
        """
        return {}

    def create_name_gen(self):
        # TODO take into account the parents!!!
        return name_gen(self.name)

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


class InputPlaceholder(Node):
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

    def __call__(self, input, hpopt_config={}):
        raise ValueError('Placeholders are not executable')


def input_key(key=None):
    node = InputPlaceholder(key, match_all_inputs=False)
    Graph.get_default().add_node(node)
    return node


def input_all():
    #TODO generate just one node placeholder_all per graph
    node = InputPlaceholder(key=None, match_all_inputs=True)
    Graph.get_default().add_node(node)
    return node


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


def merge(mode):
    node = Merge(name=None, mode=mode)
    Graph.get_default().add_node(node)
    return node


class NodeId:
    # TODO include "graph id" (to be decided) although the id can be either fully qualified and relative
    def __init__(self, name):
        if not isinstance(name, str):
            raise ValueError()
        self._name = name

    @property
    def name(self) -> str:
        return self._name


def node(node):
    # TODO Adapt a graph to node, in the future a graph won't be a node anymore?

    node = Graph.adapt(Graph.expand(node))
    #if isinstance(node, str):
    #    node = NodePlaceholder(reference_name=node)
    if isinstance(node, Node):
        Graph.get_default().add_node(node)
        return node

    raise ValueError()
    #return NodeId(node)


def node_ref(node):
    if isinstance(node, str):
        return NodeId(node)
    if isinstance(node, NodeId):
        return node
    if isinstance(node, Node):
        Graph.get_default().add_node(node)
        return NodeId(node.name)
    raise ValueError()


class GetKeys(Node):
    """
    Gets elements from lists or dictionaries
    """

    def __init__(self, name=None, keys=None, output_type=None):
        if not isinstance(keys, list):
            raise ValueError()
        if not(output_type in [None, 'd']):
            raise ValueError()
        self.keys = keys
        self.output_type = output_type
        super(GetKeys, self).__init__(name)

    @staticmethod
    def deserializer(keys, output_type=None):
        return GetKeys(name=None, keys=keys, output_type=output_type)

    def __call__(self, input, hpopt_config={}):
        if self.output_type == 'd':
            values = [input[k] for k in self.keys]
            return dict(zip(self.keys, values))

        return [input[k] for k in self.keys]


def get_keys(keys, output_type=None):
    node = GetKeys(name=None, keys=keys, output_type=output_type)
    Graph.get_default().add_node(node)
    return node


def input_keys(keys, output_type=None):
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


def dump(name=None):
    node = Dump(name)
    Graph.get_default().add_node(node)
    return node


class Switch(Node):
    """
    A node that switches between multiple inputs
    """

    # TODO define the actor that will move the switch
    def __init__(self, name=None, default_choice=None):
        #TODO allow different probabilities for different inputs
        self.default_choice = default_choice
        super().__init__(name)

    @staticmethod
    def deserializer(name):
        return Switch(name)

    def get_hpopt_config_ranges(self):
        g = self.parent
        assert g is not None
        input_binding = g.get_node_input_binding(node)
        if input_binding is None:
            return {}

        h = HPOptHelpers(self.name)
        if isinstance(input_binding, list):
            return dict([h.def_choice('options', list(range(len(input_binding))))])

        if isinstance(input_binding, list):
            return dict([h.def_choice('options', list(input_binding.keys()))])

        raise ValueError()

    def get_input_binding(self, hpopt_config={}):
        choice = HPOptHelpers(self.name, hpopt_config).get_choice('options', self.default_choice)
        if choice is None:
            return None

        g = self.parent
        assert g is not None
        input_binding = g.get_node_input_binding(node)
        assert input_binding is not None
        return input_binding[choice]

    def __call__(self, input, hpopt_config={}):
        # the selection is performed in the get_input_binding so here we simply return the input
        return input


def switch(name=None, default_choice=None) -> Node:
    node = Switch(name, default_choice)
    Graph.get_default().add_node(node)
    return node


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


def add_event_handler(event, deps):
    if deps is None:
        raise ValueError()

    graph = Graph.get_default()

    deps = get_nodes_from_struct(Graph.expand(deps))
    for n in deps:
        graph.add_node_if_obj(n)

    graph.add_event_handler(event, substitute_nodes(deps, Node.nodeId))


class Lambda(Node):
    def __init__(self, name=None, func=None):
        if not callable(func):
            raise ValueError("Param func should be a callable")
        self.func = func
        super(Lambda, self).__init__(name)

    def __call__(self, input, hpopt_config={}):
        return self.func(input)


def lambda_(f) -> Node:
    """
    Creates a lambda node. Note it is also useful to create callbacks by behaving like an identity.
    Another possibility is using the node function: node(lambda x: ...)
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


def substitute_nodes(nodes, lookup_fn):
    if isinstance(nodes, list):
        output = []
        for obj in nodes:
            if isinstance(obj, (NodeId, Node)):
                output.append(lookup_fn(obj))
            elif isinstance(obj, (list, dict)):
                output.append(substitute_nodes(obj, lookup_fn))
            else:
                output.append(obj)
        return output
    elif isinstance(nodes, dict):
        output = {}
        for key, obj in nodes.items():
            if isinstance(obj, (NodeId, Node)):
                output[key] = lookup_fn(obj)
            elif isinstance(obj, (list, dict)):
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
        elif isinstance(obj, list):
            get_nodes_from_struct(obj, output)
        elif isinstance(obj, dict):
            get_nodes_from_struct(obj, output)

    if isinstance(iterable, list):
        for obj in iterable:
            op(obj)
    elif isinstance(iterable, dict):
        for obj in iterable.values():
            op(obj)
    elif isinstance(iterable, (Node, NodeId)):
        output.append(iterable)

    return output


class Variable(Node):
    def __init__(self, name, dtypes=None):       # TODO initial value!
        self._dtypes = dtypes
        super(Variable, self).__init__(name)

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


# TODO class Subgraph(Node) the call node() will adapt a graph to a node through Subgraph


class GraphCallback:
    def on_node_execution_end(self, ctx: ExecutionContext, graph, **args):
        pass

    def on_node_add(self, graph, node: Node, **args):
        pass


class Graph(Node):
    _default = None

    #TODO serializer & deserializer

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
        # TODO specifiy parallel=True|False, that is whether the graph nodes are allowed to
        # be executed in parallel
        self.nodes = {}    # key is the node name, value is a node object
        self.links = {}    # key is the node name, value is an input binding
        self.deps = {}     # key is the node name, value is a list of node names
        self.event_handlers = {}    # key is event name, value is a node object
        self.default_output = default_output
        self.callback = callback
        super().__init__(name)

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

        if isinstance(node, Node):
            # TODO check whether node is instance of InputPlaceholder
            # raise ValueError('Input binding not supported for input placeholders')
            pass

        node = Node.get_name(node)
        assert isinstance(node, str)
        self.links[node] = copy.deepcopy(input_binding)

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

    def add_node(self, node):
        """
        Adds a node and the relative input bindings to the graph
        :param node: An object of type Node
        :param input_binding: A string, a list of strings or a dictionary representing the binding of the input of this node
        with respect to the output of another node(s)
        :return:
        """

        if node is self:
            raise ValueError('A graph cannot be a child of itself')

        if isinstance(node, Node):
            if node.parent is not None:
                if node.parent != self:
                    raise ValueError('The node is already part of another Graph')
                else:
                    return

            if node.name in self.nodes:
                raise ValueError('Duplicate node name error')
            self.nodes[node.name] = node
            node._parent_wref = weakref.ref(self)

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

    def get_node_input_binding(self, node):
        node = Node.get_name(node)
        input_binding = self.links.get(node)
        if input_binding is not None:
            input_binding = copy.deepcopy(input_binding)
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

        def run_node(self, node, hpopt_config={}):
            """
            Executes one node (after the execution of its requirements) and store the output in node_output_map
            :param node: Can be an identifier or an object of class Node
            :param hpopt_config: the hyper-parameters selected by hyperopt
            :return:
            """

            parent = self.parent

            def run_deps(node):
                for d in parent.get_node_deps(node):
                    # TODO in the future these will be executed in parallel
                    self.run_node(d, hpopt_config)

            node = parent.get_node(node)
            # use node_name instead of node.name because the pleceholder may change the current node
            # the same applies to input_binding
            node_name = node.name
            input_binding = node.get_input_binding(hpopt_config)
            run_deps(node)

            changed = True
            while changed:
                changed = False

                if node_name in self.node_output_map:
                    # this node has already been processed
                    return

                #if isinstance(node, NodePlaceholder):
                #    changed = True
                #    node = parent.get_node(node.reference_name)
                #    run_deps(node)

                if isinstance(node, InputPlaceholder):
                    # TODO if deps != [] emit a warning
                    self.node_output_map[node_name] = node.select_input(self.graph_input)
                    self._callback_on_node_exec(node)
                    return

            if input_binding is None:
                # the node doesn't have an input
                self.node_output_map[node_name] = node(None, hpopt_config)
                return

            requirements = get_nodes_from_struct(input_binding)
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
            self.node_output_map[node_name] = node(substitute_nodes(input_binding, lookup_fn=self.lookup_output),
                                                        hpopt_config=hpopt_config)
            self._callback_on_node_exec(node)

        def run_event_handlers(self, event, hpopt_config):
            handlers = self.parent.event_handlers.get(event)
            if handlers is not None:
                for h in handlers:
                    self.run_node(h, hpopt_config)

        def run(self, nodes, hpopt_config={}):
            # Execute on enter handlers
            self.run_event_handlers('enter', hpopt_config)

            for req in get_nodes_from_struct(nodes):
                self.run_node(req, hpopt_config)
            output = substitute_nodes(nodes, self.lookup_output)

            # Execute on exit handlers
            self.run_event_handlers('exit', hpopt_config)

            return output

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
        return vm.run(obj)

    @classmethod
    def adapt(cls, obj):
        adapter = cls.adapters.get(type(obj))   # TODO check key using issubclass()
        if adapter:
            return adapter(obj)
        return obj


def return_(collection, deps=None):
    """
    Sets the default output of the current graph
    :param collection: A node or an identifier or a list or dictionary of the same
    :return:
    """

    output = link(identity(), collection, deps=deps)
    Graph.get_default().default_output = output


Graph.operators_reg = {
    '#$#g.dump': Dump.deserializer,
    '#$#g.input': InputPlaceholder.deserializer,
    '#$#g.input_all': partial(InputPlaceholder.deserializer, match_all_inputs=True),
    '#$#g.identity': Identity.deserializer,
    '#$#g.switch': Switch.deserializer,
    '#$#g.return': return_,
    '#$#g.get_keys': get_keys,
    '#$#g.input_keys': input_keys,
    '#$#g.merge': merge
}

Graph.event_types = {'enter', 'exit'}   # TODO exception handler

Graph.adapters = {  # A map type: adapter
    types.FunctionType: lambda_,
    # TODO Graph: Subgraph?
}
