
import os
import time


from pg_amcd.provenance import compute_file_sha256, is_output_stale, compute_run_id


def test_sha256_deterministic(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    assert compute_file_sha256(str(p)) == compute_file_sha256(str(p))


def test_sha256_changes_with_content(tmp_path):
    a = tmp_path / "a.bin"
    a.write_bytes(b"1")
    b = tmp_path / "b.bin"
    b.write_bytes(b"2")
    assert compute_file_sha256(str(a)) != compute_file_sha256(str(b))


def test_is_output_stale_missing(tmp_path):
    inp = tmp_path / "in.mat"
    inp.write_bytes(b"d")
    assert is_output_stale(str(inp), [str(tmp_path / "missing")]) is True


def test_is_output_stale_fresh(tmp_path):
    inp = tmp_path / "in.mat"
    inp.write_bytes(b"d")
    out = tmp_path / "o.npz"
    out.write_bytes(b"x")
    older = time.time() - 10
    os.utime(str(inp), (older, older))
    os.utime(str(out), (time.time(), time.time()))
    assert is_output_stale(str(inp), [str(out)]) is False


def test_compute_run_id_deterministic():
    a = compute_run_id("c", "g", ["a", "b"])
    b = compute_run_id("c", "g", ["a", "b"])
    assert a == b and isinstance(a, str) and len(a) > 0
