"""交差したランダム切片の混合効果モデルで、水準ごとの値を求める処理のテスト。"""

import warnings

import numpy as np
import pandas
import pytest
import statsmodels.formula.api as formulas

from lexical_acquisition_mechanisms.mixed_effects_curves import (
    fit_levels,
    generalized_least_squares,
)

TRUTH = np.linspace(0.0, 1.0, 8)


def latin_square(groups: int, missing_run: bool) -> pandas.DataFrame:
    """8つの run・8つの水準のラテン方格（1組に8つの単位）。値は本当の曲線に、単位・名詞・run ごとの
    ずれと誤差を足したもの。run 0 は値が 0.5 大きい。

    missing_run なら、run 0 の低い4つの水準の行を除く（run 0 が、高い水準にだけ偏る）。
    """
    random_numbers = np.random.default_rng(0)
    units = random_numbers.normal(0, 0.15, groups * 8)
    nouns = random_numbers.normal(0, 0.1, groups * 8)
    rows = []
    for group in range(groups):
        for run in range(8):
            for row in range(8):
                level = (row + run) % 8
                if missing_run and run == 0 and level < 4:
                    continue
                unit = group * 8 + row
                noun = group * 8 + (2 * row + run) % 8
                value = TRUTH[level] + units[unit] + nouns[noun] + (0.5 if run == 0 else 0.0)
                rows.append(
                    {
                        "unit": unit,
                        "noun": noun,
                        "run": run,
                        "level": level,
                        "value": value + random_numbers.normal(0, 0.05),
                    }
                )
    return pandas.DataFrame(rows)


def fit(frame: pandas.DataFrame):
    intercepts = [frame[name].to_numpy() for name in ("unit", "noun", "run")]
    return fit_levels(frame["level"].to_numpy(), frame["value"].to_numpy(), intercepts, 8)


def test_estimates_agree_with_statsmodels() -> None:
    frame = latin_square(groups=2, missing_run=True)

    result = fit(frame)

    model = formulas.mixedlm(
        "value ~ 0 + C(level)",
        frame,
        groups=np.ones(len(frame)),
        re_formula="0",
        vc_formula={name: f"0 + C({name})" for name in ("unit", "noun", "run")},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reference = model.fit(reml=True)
    assert result.values == pytest.approx(reference.fe_params.to_numpy(), abs=1e-3)
    assert result.standard_errors == pytest.approx(reference.bse_fe.to_numpy(), rel=1e-2)
    variances = dict(zip(model.exog_vc.names, reference.vcomp, strict=True))
    expected = [variances[name] for name in ("unit", "noun", "run")]
    assert result.intercept_variances == pytest.approx(expected, rel=2e-2, abs=1e-4)
    assert result.residual_variance == pytest.approx(reference.scale, rel=1e-2)


def test_curve_shape_is_kept_when_a_run_covers_only_high_levels() -> None:
    frame = latin_square(groups=9, missing_run=True)

    result = fit(frame)

    # 全体のずれを除いた形を比べる。水準ごとの平均は、run 0 の 0.5 のずれが高い水準にだけ入って
    # 形が崩れるが、混合効果モデルは崩れない
    means = frame.groupby("level")["value"].mean().to_numpy()
    assert np.max(np.abs(means - np.mean(means - TRUTH) - TRUTH)) > 0.03
    assert np.max(np.abs(result.values - np.mean(result.values - TRUTH) - TRUTH)) < 0.015


def test_least_squares_with_the_fitted_variances_reproduces_the_fit() -> None:
    frame = latin_square(groups=3, missing_run=True)
    result = fit(frame)
    intercepts = [frame[name].to_numpy() for name in ("unit", "noun", "run")]

    values = generalized_least_squares(
        frame["level"].to_numpy(),
        frame["value"].to_numpy(),
        intercepts,
        result.intercept_variances,
        result.residual_variance,
        8,
    )

    assert values == pytest.approx(result.values, abs=1e-8)


def test_values_without_spread_fall_back_to_level_means() -> None:
    levels = np.array([0, 0, 1, 1])
    targets = np.array([0.2, 0.2, 0.2, 0.2])

    result = fit_levels(levels, targets, [np.array([0, 1, 0, 1])], 3)

    assert result.intercept_variances is None
    assert result.values[:2] == pytest.approx([0.2, 0.2])
    assert np.isnan(result.values[2])
