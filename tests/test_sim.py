from dataclasses import replace

import numpy as np
import pytest

from invsim import Config, Policy, chen_bullwhip_lower_bound, newsvendor_z, simulate


def test_flow_conservation():
    r = simulate(Config(days=300))
    assert np.all(r.sales <= r.demand + 1e-9)
    assert np.all(r.store_on_hand >= -1e-9)
    assert np.all(r.dc_on_hand >= -1e-9)
    assert 0 < r.fill_rate <= 1


def test_more_safety_stock_means_better_service():
    lo = simulate(Config(days=400, store=Policy(z=0.0)))
    hi = simulate(Config(days=400, store=Policy(z=2.5)))
    assert hi.fill_rate > lo.fill_rate
    assert hi.store_on_hand.mean() > lo.store_on_hand.mean()


@pytest.mark.parametrize("window", [3, 7, 14])
def test_bullwhip_matches_chen_et_al(window):
    """Single stage, i.i.d. demand, backorders, z = 0, returns allowed: the
    textbook setting where the Chen et al. expression holds with equality."""
    vals = []
    for seed in range(3):
        cfg = Config(days=4000, weekly=(1,) * 7, lost_sales=False, infinite_dc=True,
                     allow_returns=True, seed=seed,
                     store=Policy(z=0, forecast="moving_average", window=window))
        vals.append(simulate(cfg).bullwhip()["store_orders"])
    lead_periods = cfg.lead_store + cfg.store.review_days - 1
    assert np.mean(vals) == pytest.approx(chen_bullwhip_lower_bound(lead_periods, window), rel=0.05)


def test_static_base_stock_passes_demand_through():
    cfg = Config(days=2000, lost_sales=False, infinite_dc=True, store=Policy(z=1.0))
    assert simulate(cfg).bullwhip()["store_orders"] == pytest.approx(1.0, abs=0.05)


def test_pos_sharing_reduces_upstream_bullwhip():
    pol = Policy(forecast="moving_average", window=5)
    base = Config(days=730, store=pol, dc=replace(pol, z=2.0))
    no = simulate(base).bullwhip()["dc_orders"]
    yes = simulate(replace(base, dc_uses_pos=True)).bullwhip()["dc_orders"]
    assert yes < no


def test_sS_orders_less_often_than_base_stock():
    K = 150.0
    bs = simulate(Config(days=400, store=Policy(order_cost=K)))
    ss = simulate(Config(days=400, store=Policy(kind="sS", order_cost=K)))
    assert (ss.store_orders > 0).sum() < (bs.store_orders > 0).sum()
    assert ss.costs["ordering"] < bs.costs["ordering"]


def test_newsvendor_z():
    assert newsvendor_z(1.0, 1.0) == pytest.approx(0.0)
    assert newsvendor_z(0.3, 8.0) == pytest.approx(1.80, abs=0.01)
