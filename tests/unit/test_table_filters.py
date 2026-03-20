from bench.table_filters import filter_block_height, filter_rows, wrap_text


def test_filter_rows_case_insensitive_substring():
    rows = [{"engine": "manticore"}, {"engine": "qdrant"}]
    out = filter_rows(rows, {"engine": "tic"}, lambda row, col: row.get(col, ""))
    assert out == [{"engine": "manticore"}]


def test_filter_block_height_wraps_long_filter():
    widths = {"model": 8}
    filters = {"model": "openai text embedding"}
    assert filter_block_height(filters, widths) == len(wrap_text(filters["model"], 8))
