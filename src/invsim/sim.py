"""Daily discrete-time simulation of a two-echelon distribution network.

    plant (unlimited) --L_dc days--> DC --L_store days--> N stores --> customers

Sequence within each day t
    1. shipments due today arrive (stores, DC)
    2. stores face demand; unmet demand is lost (or backordered)
    3. stores review inventory position and order from the DC
    4. DC fills store orders (rationed proportionally if short), ships them
    5. DC reviews and orders from the plant
    6. holding, shortage and fixed ordering costs are booked
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Policy:
    kind: str = "base_stock"      # "base_stock" (order-up-to S) or "sS"
    z: float = 1.65               # safety factor
    review_days: int = 1          # R: order only every R days
    forecast: str = "static"      # "static" (true mean/sd) or "moving_average"
    window: int = 7               # moving-average length p
    order_cost: float = 0.0       # K, per order placed (used for EOQ in (s,S))


@dataclass(frozen=True)
class Config:
    n_stores: int = 5
    days: int = 730
    warmup: int = 60
    store_mean: tuple = (40, 55, 30, 70, 45)
    cv: float = 0.35                       # demand coefficient of variation
    weekly: tuple = (0.85, 0.85, 0.9, 0.95, 1.1, 1.3, 1.05)
    lead_store: int = 2
    lead_dc: int = 6
    store: Policy = Policy()
    dc: Policy = Policy(z=2.0)
    dc_uses_pos: bool = False              # information sharing: DC forecasts from POS
    lost_sales: bool = True
    infinite_dc: bool = False              # DC always ships in full (single-echelon view)
    allow_returns: bool = False            # negative orders (textbook bullwhip setting)
    h_store: float = 0.30                  # INR / unit / day
    h_dc: float = 0.12
    shortage: float = 8.0                  # INR per unit of lost / backordered demand
    seed: int = 0


@dataclass
class SimResult:
    config: Config
    demand: np.ndarray          # (T, N)
    sales: np.ndarray           # (T, N)
    store_orders: np.ndarray    # (T, N)
    dc_orders: np.ndarray       # (T,)
    store_on_hand: np.ndarray   # (T, N)
    dc_on_hand: np.ndarray      # (T,)
    costs: dict = field(default_factory=dict)

    def _w(self, a):
        return a[self.config.warmup:]

    @property
    def fill_rate(self) -> float:
        return float(self._w(self.sales).sum() / self._w(self.demand).sum())

    def bullwhip(self) -> dict:
        """Variance amplification relative to aggregate customer demand."""
        d = self._w(self.demand).sum(1).var()
        return {"store_orders": float(self._w(self.store_orders).sum(1).var() / d),
                "dc_orders": float(self._w(self.dc_orders).var() / d)}

    def summary(self) -> dict:
        days = self.config.days - self.config.warmup
        per_day = {k: float(v) / days for k, v in self.costs.items()}
        return {"fill_rate_pct": round(100 * self.fill_rate, 2),
                **{f"{k}_per_day": round(v, 1) for k, v in per_day.items()},
                "total_cost_per_day": round(float(sum(per_day.values())), 1),
                "avg_store_inventory": round(float(self._w(self.store_on_hand).sum(1).mean()), 1),
                "avg_dc_inventory": round(float(self._w(self.dc_on_hand).mean()), 1),
                **{f"bullwhip_{k}": round(v, 2) for k, v in self.bullwhip().items()}}


def _levels(pol: Policy, mean, sd, lead, h=0.30):
    """(reorder point s, order-up-to S).

    An order placed at the end of day t arrives at the start of day t + lead,
    and the next order after it arrives R days later, so the inventory
    position must protect against demand over lead + R - 1 days."""
    cover = max(lead + pol.review_days - 1, 1)
    base = mean * cover + pol.z * sd * np.sqrt(cover)
    if pol.kind == "base_stock":
        return base, base
    if pol.kind == "sS":
        eoq = np.sqrt(2 * pol.order_cost * np.maximum(mean, 1e-9) / h)
        return base, base + eoq
    raise ValueError(pol.kind)


class _Estimator:
    def __init__(self, pol: Policy, true_mean, true_sd):
        self.pol, self.mean, self.sd = pol, np.asarray(true_mean, float), np.asarray(true_sd, float)
        self.hist: list[np.ndarray] = []

    def update(self, x):
        if self.pol.forecast == "static":
            return
        self.hist.append(np.asarray(x, float))
        if len(self.hist) >= self.pol.window:
            h = np.array(self.hist[-self.pol.window:])
            self.mean = h.mean(0)
            if self.pol.window > 1:
                self.sd = h.std(0, ddof=1)


def simulate(cfg: Config) -> SimResult:
    rng = np.random.default_rng(cfg.seed)
    N, T = cfg.n_stores, cfg.days
    mu = np.array(cfg.store_mean[:N], float)
    wk = np.array(cfg.weekly) / np.mean(cfg.weekly)

    # negative-binomial demand (gamma-Poisson) with weekly seasonality
    lam = mu[None, :] * wk[np.arange(T) % 7][:, None]
    var = (cfg.cv * lam) ** 2 + lam
    shape = lam ** 2 / np.maximum(var - lam, 1e-9)
    demand = rng.poisson(rng.gamma(shape, lam / shape)).astype(float)

    sd_daily = np.sqrt((cfg.cv * mu) ** 2 + mu + (mu * wk.std()) ** 2)
    st_est = _Estimator(cfg.store, mu, sd_daily)
    dc_mean = mu.sum()
    dc_sd = np.sqrt((sd_daily ** 2).sum())
    dc_est = _Estimator(cfg.dc, dc_mean, dc_sd)

    s0, S0 = _levels(cfg.store, mu, sd_daily, cfg.lead_store)
    on_hand = S0.copy()
    dc_oh = float(_levels(cfg.dc, dc_mean, dc_sd, cfg.lead_dc)[1])
    store_arr = np.zeros((T + cfg.lead_store + 1, N))
    dc_arr = np.zeros(T + cfg.lead_dc + 1)
    owed = np.zeros(N)                 # DC backlog per store
    backlog = np.zeros(N)              # customer backorders (if not lost sales)

    out = {k: np.zeros((T, N)) for k in ("sales", "orders", "oh")}
    dc_orders, dc_ohs = np.zeros(T), np.zeros(T)
    cost = {"holding_store": 0.0, "holding_dc": 0.0, "shortage": 0.0, "ordering": 0.0}

    for t in range(T):
        on_hand += store_arr[t]
        dc_oh += dc_arr[t]

        need = demand[t] + backlog
        sold = np.minimum(on_hand, need)
        on_hand -= sold
        short = need - sold
        backlog = np.zeros(N) if cfg.lost_sales else short
        sales_today = np.minimum(sold, demand[t]) if not cfg.lost_sales else sold

        st_est.update(demand[t])
        q = np.zeros(N)
        if t % cfg.store.review_days == 0:
            s, S = _levels(cfg.store, st_est.mean, st_est.sd, cfg.lead_store, cfg.h_store)
            ip = on_hand + store_arr[t + 1:].sum(0) + owed - backlog
            if cfg.allow_returns:
                q = S - ip
            else:
                q = np.where(ip <= s, np.maximum(S - ip, 0), 0)
        owed += q

        if cfg.infinite_dc:
            ship = owed.copy()
        else:
            tot = owed.sum()
            ship = owed * min(1.0, dc_oh / tot) if tot > 0 else owed * 0
            dc_oh -= ship.sum()
        owed -= ship
        store_arr[t + cfg.lead_store] += ship

        dc_est.update(demand[t].sum() if cfg.dc_uses_pos else q.sum())
        dq = 0.0
        if t % cfg.dc.review_days == 0 and not cfg.infinite_dc:
            s, S = _levels(cfg.dc, dc_est.mean, dc_est.sd, cfg.lead_dc, cfg.h_dc)
            ip = dc_oh + dc_arr[t + 1:].sum() - owed.sum()
            if ip <= s:
                dq = max(S - ip, 0.0)
        dc_arr[t + cfg.lead_dc] += dq

        if t >= cfg.warmup:
            cost["holding_store"] += cfg.h_store * on_hand.sum()
            cost["holding_dc"] += cfg.h_dc * dc_oh
            cost["shortage"] += cfg.shortage * short.sum()
            cost["ordering"] += cfg.store.order_cost * float((q > 0).sum()) + cfg.dc.order_cost * float(dq > 0)

        out["sales"][t], out["orders"][t], out["oh"][t] = sales_today, q, on_hand
        dc_orders[t], dc_ohs[t] = dq, dc_oh

    return SimResult(cfg, demand, out["sales"], out["orders"], dc_orders, out["oh"], dc_ohs, cost)


def replicate(cfg: Config, n: int = 5) -> dict:
    """Mean and standard deviation of every summary metric over n seeds."""
    runs = [simulate(replace(cfg, seed=cfg.seed + k)).summary() for k in range(n)]
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs])))
            for k in runs[0]}
