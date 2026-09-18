from tgaoi.ir import Graph, Node, TensorType


def test_graph_validation():
    g = Graph("add", {"a": TensorType((2,)), "b": TensorType((2,))}, ["c"])
    g.add(Node("c", "ADD", ["a", "b"], output_type=TensorType((2,))))
    g.validate()


def test_missing_input_fails():
    g = Graph("bad", {}, ["y"], [Node("y", "ADD", ["missing"])])
    try:
        g.validate()
    except ValueError:
        return
    raise AssertionError("expected validation failure")
