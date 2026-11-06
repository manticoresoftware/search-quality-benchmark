from pathlib import Path

from bench.config import AppConfig, ManticoreConfig
from bench.manticore_snapshot import dump_tables, restore_dump


def test_dump_tables_invokes_mysqldump(monkeypatch, tmp_path: Path):
    cfg = AppConfig(manticore=ManticoreConfig(host="127.0.0.1", sql_port=9306, user="", password=""))
    called = {}

    def fake_run(cmd, stdout=None, stderr=None, check=None, env=None, stdin=None):
        called["cmd"] = cmd
        if stdout is not None:
            stdout.write(b"-- dump\n")
        return None

    monkeypatch.setattr("bench.manticore_snapshot.subprocess.run", fake_run)
    out = tmp_path / "dump.sql"
    result = dump_tables(cfg, mode="both", output_path=str(out))
    assert out.exists()
    assert called["cmd"][0] == "mysqldump"
    assert "--no-tablespaces" in called["cmd"]
    assert "Manticore" in called["cmd"]
    assert "trec_covid_ft" in called["cmd"]
    assert "trec_covid_vec" in called["cmd"]
    assert result.action == "dump"


def test_restore_dump_invokes_mysql(monkeypatch, tmp_path: Path):
    cfg = AppConfig(manticore=ManticoreConfig(host="127.0.0.1", sql_port=9306, user="", password=""))
    called = {}
    inp = tmp_path / "dump.sql"
    inp.write_text("SELECT 1;\n", encoding="utf-8")

    def fake_run(cmd, stdout=None, stderr=None, check=None, env=None, stdin=None):
        called["cmd"] = cmd
        called["has_stdin"] = stdin is not None
        return None

    monkeypatch.setattr("bench.manticore_snapshot.subprocess.run", fake_run)
    result = restore_dump(cfg, input_path=str(inp))
    assert called["cmd"][0] == "mysql"
    assert called["has_stdin"] is True
    assert result.action == "restore"
