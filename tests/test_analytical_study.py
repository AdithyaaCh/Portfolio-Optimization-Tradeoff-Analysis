import numpy as np
import pandas as pd
import pytest

from tvmvp_ext.analytical_study import empirical_es, lagged_high_volatility, summary
from tvmvp_ext.covariance import downside_second_moment, sample_factor_covariance


def test_es_uses_fractional_tail_mass():
    # Worst 1.5 observations: (10 + 0.5*4) / 1.5 = 8.
    assert empirical_es(np.array([0., 1., 2., 3., 4., 10.]), .75) == pytest.approx(8.)
    assert empirical_es(np.array([0., 1., 2., 3., 4., 10.]), .95) == pytest.approx(10.)
    assert empirical_es(np.array([2., 2., 2.]), .8) == pytest.approx(2.)


def test_cost_accounting_and_gross_net_metrics_are_distinct():
    dates = pd.bdate_range("2020-01-01", periods=4)
    gross = pd.Series([.01, -.02, .005, .004], index=dates)
    turnover = pd.Series([.3, 0., .1, 0.], index=dates)
    net = gross - .001 * 2 * turnover
    result = summary(gross, net, turnover)
    assert result["Annual arithmetic cost drag"] == pytest.approx(.001 * 2 * .4 * 252 / 4)
    assert result["Net daily ES95"] >= result["Gross daily ES95"]
    with pytest.raises(ValueError):
        summary(gross, net.iloc[::-1], turnover)


def test_existing_downside_estimator_depends_on_arbitrary_factor_sign():
    # This records a discovered limitation, not a desired property of a valid estimator.
    f = np.array([[-4.], [1.], [1.], [1.], [1.]])
    b = np.array([[1.], [2.]])
    c, _ = sample_factor_covariance(f, 1e-9)
    cf, _ = sample_factor_covariance(-f, 1e-9)
    d, _ = downside_second_moment(f, 1e-9)
    df, _ = downside_second_moment(-f, 1e-9)
    assert np.allclose(f @ b.T, (-f) @ (-b).T)
    assert np.allclose(b @ c @ b.T, (-b) @ cf @ (-b).T)
    assert not np.allclose(b @ d @ b.T, (-b) @ df @ (-b).T)


def test_volatility_state_uses_only_prior_returns():
    dates = pd.bdate_range("2020-01-01", periods=60)
    values = pd.Series(np.zeros(60), index=dates)
    config = {"evaluation": {"index_volatility_days": 5, "high_volatility_min_history": 10,
                              "high_volatility_quantile": .75}}
    baseline = lagged_high_volatility(values, config)
    changed = values.copy(); changed.iloc[-1] = 1
    revised = lagged_high_volatility(changed, config)
    assert baseline.iloc[-1] == revised.iloc[-1]
