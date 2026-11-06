from pathlib import Path

from bench.runbench import _load_qrels_map


def test_qrels_parsing(tmp_path: Path):
    p = tmp_path / "qrels.tsv"
    p.write_text(
        "topic_id\tdoc_id\trelevance\n"
        "1\td1\t2\n"
        "1\td2\t0\n"
        "2\td3\t1\n",
        encoding="utf-8",
    )
    parsed = _load_qrels_map(p)
    assert parsed["1"]["d1"] == 2
    assert parsed["1"]["d2"] == 0
    assert parsed["2"]["d3"] == 1
