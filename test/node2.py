import hypergraph.graph as hg

# TODO Multi-layered graph idea. Some nodes are present in multiple graphs and they can influence the behaviour
# they have in different graphs

# !!!!!!!!
# TODO special prefix for node names identifiers so that we can generate more complicate structures such as:
# [['#@#node1', '#@#node2'], {'abc': '#@#node3'}
# !!!!!!!!

graph2 = hg.Graph()
with graph2.as_default():
    hg.return_(hg.static(100))

graph1 = hg.Graph()
with graph1.as_default():
    hg.link(hg.mark('list1'), {'v': hg.static(list(range(5)))})
    a = hg.link(hg.merge('d'), [hg.get_input_keys(['nd1', 'nd2'], 'd'), 'list1'])
    b = hg.link(lambda x: x ** 2, hg.placeholder('nd2'))

    c = hg.link(hg.switch('sw1', 1), [hg.static(111), hg.static(222)])
    c = hg.link(hg.switch('sw1', 1), [111, 222])

    # TODO map node to a list or dict from input... eg foreach? act on the graph?
    # foreach iterates through items of a dictionary or list and executes another node...

    # TODO node get_keys with input: {'collection': data, 'key': node}... can simulate a switch...

    hg.return_([a, b, graph2, c])

    #return_(link(get_keys({'a': 'nd1', 'b': 'nd2'}), placeholder_all()))

    #return_(link(get_keys(['nd1', 'nd2']), placeholder_all()))
    #link(add_node(graph2), placeholder_all())

ctx = hg.ExecutionContext()
with ctx.as_default():
    print(graph1(input={'nd1': 1, 'nd2': 5, 'nd3': 6}))
