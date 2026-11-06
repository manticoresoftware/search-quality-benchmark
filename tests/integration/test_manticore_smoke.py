import os

import pytest

from bench.config import AppConfig
from bench.manticore import ManticoreClient


pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.getenv("MANTICORE_INTEGRATION") != "1", reason="integration test disabled")
def test_manticore_fulltext_and_vector_smoke():
    cfg = AppConfig()
    client = ManticoreClient(cfg.manticore)
    try:
        client.drop_table_if_exists("bench_it_ft")
        client.drop_table_if_exists("bench_it_vec")
        client.create_fulltext_table("bench_it_ft")
        client.create_vector_table(
            table_name="bench_it_vec",
            vector_field="embedding",
            model_name=cfg.manticore.vector_model,
            vector_from=cfg.manticore.vector_from,
            api_key=cfg.manticore.vector_api_key,
            api_key_file=cfg.manticore.vector_api_key_file,
        )
        docs = [
            {"id": 1, "doc_id": "d1", "title": "covid treatment", "abstract": "trial data", "doi": "", "date": ""},
            {"id": 2, "doc_id": "d2", "title": "virus model", "abstract": "epidemic forecast", "doi": "", "date": ""},
        ]
        stats_ft = client.bulk_insert_docs("bench_it_ft", docs, batch_size=2, retries=1)
        stats_vec = client.bulk_insert_docs("bench_it_vec", docs, batch_size=2, retries=1)
        assert stats_ft.rows_inserted == 2
        assert stats_vec.rows_inserted == 2

        rows_ft = client.query("SELECT doc_id FROM bench_it_ft WHERE MATCH(%s) LIMIT 5", ("covid",))
        assert len(rows_ft) >= 1
        rows_vec = client.query("SELECT doc_id FROM bench_it_vec WHERE knn(embedding, 2, %s) LIMIT 2", ("covid treatment",))
        assert len(rows_vec) >= 1
    finally:
        client.drop_table_if_exists("bench_it_ft")
        client.drop_table_if_exists("bench_it_vec")
        client.close()
