from bench.runbench import _hybrid_implementation


def test_new_run_artifacts_classify_hybrid_execution_location() -> None:
    assert _hybrid_implementation("qdrant", "hybrid") == "native"
    assert _hybrid_implementation("milvus", "hybrid") == "native"
    assert _hybrid_implementation("mariadb", "hybrid") == "native"
    assert _hybrid_implementation("serenedb", "hybrid") == "non-native"
    assert _hybrid_implementation("qdrant", "vector") == ""
