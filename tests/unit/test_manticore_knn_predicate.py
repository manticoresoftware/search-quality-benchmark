from bench.config import AppConfig
from bench.runbench import _manticore_knn_predicate


def test_knn_predicate_without_explicit_options_uses_internal_defaults():
    cfg = AppConfig()
    cfg.benchmark.limit = 100
    cfg.manticore.vector_field = "embedding"
    cfg.manticore.vector_search_oversampling = None
    cfg.manticore.vector_search_rescore = None

    pred = _manticore_knn_predicate(cfg, "(0.1,0.2)")
    assert pred == "knn(embedding, 100, (0.1,0.2))"


def test_knn_predicate_with_explicit_options_includes_option_block():
    cfg = AppConfig()
    cfg.benchmark.limit = 25
    cfg.manticore.vector_field = "vec"
    cfg.manticore.vector_search_oversampling = 3.0
    cfg.manticore.vector_search_rescore = 1

    pred = _manticore_knn_predicate(cfg, "(1,2,3)")
    assert pred == "knn(vec, 25, (1,2,3), {oversampling=3.0, rescore=1})"
