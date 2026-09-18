from tgaoi.aoi import softmax_graph, rmsnorm_graph, linear_graph, attention_graph, adam_graph


def test_starter_aois_validate():
    for g in [softmax_graph(), rmsnorm_graph(), linear_graph(), attention_graph(), adam_graph()]:
        g.validate()
        assert g.outputs
