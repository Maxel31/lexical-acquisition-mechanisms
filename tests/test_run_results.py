"""run ごとの途中の結果のテスト。"""

from pathlib import Path

import pytest

from lexical_acquisition_mechanisms.output_records import Step, run_unless_unchanged
from lexical_acquisition_mechanisms.run_results import RunResults


def make_step(tmp_path: Path, seed: int) -> Step:
    return Step("measure", inputs={}, outputs=[tmp_path / "out.csv"], settings={"seed": seed})


def test_results_are_kept_until_cleared_and_only_for_the_same_step(tmp_path: Path) -> None:
    results = RunResults(make_step(tmp_path, 0))
    results.put("run-1", [{"value": 1}])

    assert RunResults(make_step(tmp_path, 0)).get("run-1") == [{"value": 1}]
    # 設定が違う処理の結果は使わない
    assert RunResults(make_step(tmp_path, 1)).get("run-1") is None
    assert results.get("run-2") is None
    # --recompute のときは、残っている結果を使わない
    assert RunResults(make_step(tmp_path, 0), recompute=True).get("run-1") is None


def test_a_stopped_step_resumes_from_the_runs_it_finished(tmp_path: Path) -> None:
    step = make_step(tmp_path, 0)
    measured = []

    def compute(runs: list[str], failing: str = "") -> None:
        results = RunResults(step)
        rows = []
        for run in runs:
            done = results.get(run)
            if done is not None:
                rows += done
                continue
            if run == failing:
                raise RuntimeError("止まった")
            measured.append(run)
            rows.append(run)
            results.put(run, [run])
        (tmp_path / "out.csv").write_text(",".join(rows), encoding="utf-8")

    with pytest.raises(RuntimeError):
        run_unless_unchanged(step, lambda: compute(["run-1", "run-2"], failing="run-2"))
    run_unless_unchanged(step, lambda: compute(["run-1", "run-2"]))
    RunResults(step).clear()

    # run-1 は1回だけ測り、止まった後は run-2 から測る
    assert measured == ["run-1", "run-2"]
    assert (tmp_path / "out.csv").read_text(encoding="utf-8") == "run-1,run-2"
    assert not (tmp_path / "out.csv.runs").exists()
