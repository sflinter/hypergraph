import hypergraph as hg

graph1 = hg.Graph()
with graph1.as_default():
    hg.add_event_handler('enter', deps=[hg.dump() << '* graph1 enter *'])

    hg.mark('list1') << {'v': list(range(5))}
    # mark is just and identity node, it can be used to name a particular branch of the graph
    # that is mentioned in multiple input bindings or dependencies lists
    hg.mark('list2') << {'v': list(range(7))}
    a = hg.merge('d') << [hg.input_keys(['nd1', 'nd2'], 'd'), hg.node_ref('list1')]
    b = hg.node(lambda x: x ** 2) << hg.input_key('nd2')  # node() can also be used to "adapt" lambda functions to nodes
    print("b.name="+b.name)

    c = hg.switch(default=0, name='sw1') << [a, hg.node_ref(a), hg.node_ref(b)]   # note multiple ref form possible

    # TODO map node to a list or dict from input... eg foreach? act on the graph?
    # foreach iterates through items of a dictionary or list and executes another node...

    graph2 = hg.Graph()    # inner graph
    with graph2.as_default():
        hg.add_event_handler('enter', deps=[hg.dump() << '* graph2 enter *'])
        hg.mark('n1234') << [3, 4, hg.input_all()]
        hg.output() << {'t1': [1, 2, hg.node_ref('n1234')], 't2': hg.input_all()} << hg.deps(hg.dump() << '* graph2 dump node *')
        hg.add_event_handler('exit', deps=hg.dump() << '* graph2 exit *')

    # now invoke graph2 with a certain input
    g2a = hg.node(graph2) << 5
    # and again but with a different input (but the internal status is shared, if there's any)
    g2b = hg.node(graph2) << 6

    dp1 = hg.dump() << "test test test"
    hg.mark('list3') << {'v': list(range(5))} << hg.deps([dp1, hg.node_ref(dp1)])

    hg.mark('dict1') << {'a1': {'b1': hg.node_ref('list1'), 'b2': hg.input_key('nd3'), 'b3': hg.node_ref('list2')},
                         'a2': hg.input_all()}

    ret1 = hg.mark('ret1') << {'h1': [g2a, g2b], 'h2': a}
    hg.output() << ret1['h1'][0]  # the expression node[key] is equivalent to expression link(get_key(key), node)

ctx = hg.ExecutionContext()
with ctx.as_default():
    print(graph1(input={'nd1': 1, 'nd2': 5, 'nd3': 6}))
