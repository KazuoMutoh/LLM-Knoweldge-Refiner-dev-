import pytest

from simple_active_refine.io_utils import get_iteration_dir


def test_get_iteration_dir_basic(tmp_path):
    base = tmp_path / "experiments" / "20260111" / "run"
    assert get_iteration_dir(base, 0) == base / "iter_0"
    assert get_iteration_dir(base, 3) == base / "iter_3"


def test_get_iteration_dir_rejects_negative(tmp_path):
    with pytest.raises(ValueError):
        get_iteration_dir(tmp_path, -1)
