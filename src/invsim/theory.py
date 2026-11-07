"""Closed-form results the simulation is checked against."""

from __future__ import annotations

from scipy.stats import norm


def chen_bullwhip_lower_bound(lead_periods: int, window: int) -> float:
    """Chen, Drezner, Ryan & Simchi-Levi (2000): an order-up-to policy that
    re-estimates mean demand with a p-period moving average amplifies order
    variance by at least 1 + 2L/p + 2L^2/p^2 (equality when the safety-stock
    term is constant)."""
    L, p = lead_periods, window
    return 1 + 2 * L / p + 2 * L ** 2 / p ** 2


def newsvendor_z(holding_cost: float, shortage_cost: float) -> float:
    """Safety factor at the critical ratio p / (p + h)."""
    return float(norm.ppf(shortage_cost / (shortage_cost + holding_cost)))
