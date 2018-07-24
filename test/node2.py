from hypergraph.graph import Graph, ExecutionContext, dump, switch, link, mark, input_key, input_all, Variable, get_keys, return_, input_keys, merge, node, node_ref, add_event_handler

# TODO Multi-layered graph idea. Some nodes are present in multiple graphs and they can influence the behaviour
# they have in different graphs

graph1 = Graph()
with graph1.as_default():
    add_event_handler('enter', deps=[link(dump(), '*** graph1 enter ***')])

    mark('list1') << {'v': list(range(5))}
    # mark is just and identity node, it can be used to name a particular branch of the graph
    # that is mentioned in multiple input bindings or dependencies lists
    mark('list2') << {'v': list(range(7))}
    a = merge('d') << [input_keys(['nd1', 'nd2'], 'd'), node_ref('list1')]
    b = node(lambda x: x ** 2) << input_key('nd2')  # node() can also be used to "adapt" lambda functions to nodes
    print("b.name="+b.name)

    c = switch('sw1', 0) << [a, node_ref(a), node_ref(b)]   # note multiple ref form possible

    # TODO map node to a list or dict from input... eg foreach? act on the graph?
    # foreach iterates through items of a dictionary or list and executes another node...

    graph2 = Graph()    # inner graph
    with graph2.as_default():
        add_event_handler('enter', deps=[link(dump(), '*** graph2 enter ***')])
        mark('n1234') << [3, 4, input_all()]
        # TODO create return_() pseudo node so we can do this: return_() << value
        return_({'t1': [1, 2, node_ref('n1234')], 't2': input_all()},
                deps=link(dump(), '*** graph2 dump node ***'))
        add_event_handler('exit', deps=link(dump(), '*** graph2 exit ***'))

    # now invoke graph2 with a certain input
    g2a = node(graph2) << 5
    # and again but with a different input (but the internal status is shared, if there's any)
    g2b = node(graph2) << 6

    dp1 = dump() << "test test test"
    link(mark('list3'), {'v': list(range(5))}, deps=[dp1, node_ref(dp1)])

    mark('dict1') << {'a1': {'b1': node_ref('list1'), 'b2': input_key('nd3'), 'b3': node_ref('list2')},
                      'a2': input_all()}

    ret1 = mark('ret1') << {'h1': [g2a, g2b], 'h2': a}
    return_(ret1['h1'][0])  # the expression node[key] is equivalent to expression link(get_key(key), node)

    #return_(link(get_keys({'a': 'nd1', 'b': 'nd2'}), placeholder_all()))

    #return_(link(get_keys(['nd1', 'nd2']), placeholder_all()))
    #link(add_node(graph2), placeholder_all())

ctx = ExecutionContext()
with ctx.as_default():
    print(graph1(input={'nd1': 1, 'nd2': 5, 'nd3': 6}))
