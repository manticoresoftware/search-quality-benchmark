from pathlib import Path

from bench.engines import _linux_mem_breakdown, _linux_proc_tree_pids


def _write_status(proc_root: Path, pid: int, *, ppid: int, rss_anon: int, rss_file: int, rss_shmem: int, vm_lck: int) -> None:
    proc_dir = proc_root / str(pid)
    proc_dir.mkdir(parents=True, exist_ok=True)
    (proc_dir / "status").write_text(
        "\n".join(
            [
                f"Pid:\t{pid}",
                f"PPid:\t{ppid}",
                f"RssAnon:\t{rss_anon} kB",
                f"RssFile:\t{rss_file} kB",
                f"RssShmem:\t{rss_shmem} kB",
                f"VmLck:\t{vm_lck} kB",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_linux_proc_tree_pids_includes_descendants(tmp_path: Path):
    _write_status(tmp_path, 100, ppid=1, rss_anon=1, rss_file=1, rss_shmem=0, vm_lck=0)
    _write_status(tmp_path, 101, ppid=100, rss_anon=1, rss_file=1, rss_shmem=0, vm_lck=0)
    _write_status(tmp_path, 102, ppid=101, rss_anon=1, rss_file=1, rss_shmem=0, vm_lck=0)
    _write_status(tmp_path, 200, ppid=1, rss_anon=1, rss_file=1, rss_shmem=0, vm_lck=0)

    assert _linux_proc_tree_pids(100, proc_root=tmp_path) == [100, 101, 102]


def test_linux_mem_breakdown_sums_process_tree(tmp_path: Path):
    _write_status(tmp_path, 100, ppid=1, rss_anon=1024, rss_file=2048, rss_shmem=512, vm_lck=256)
    _write_status(tmp_path, 101, ppid=100, rss_anon=2048, rss_file=1024, rss_shmem=512, vm_lck=128)
    _write_status(tmp_path, 102, ppid=101, rss_anon=1024, rss_file=0, rss_shmem=1024, vm_lck=64)

    breakdown = _linux_mem_breakdown(100, proc_root=tmp_path)

    assert breakdown == {
        "anon": "4.0M",
        "mmap": "5.0M",
        "locked": "448.0K",
    }
