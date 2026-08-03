"""
risk_management.py

Basic risk management and position sizing utilities for
quantitative trading research.
"""


def position_size(account_balance: float, risk_per_trade: float,
                   entry_price: float, stop_loss_price: float) -> float:
    """
    Calculate position size based on fixed fractional risk.

    Parameters
    ----------
    account_balance : total capital available
    risk_per_trade : fraction of account to risk (e.g. 0.01 = 1%)
    entry_price : intended entry price
    stop_loss_price : stop-loss price

    Returns
    -------
    Position size in units of the asset.
    """
    risk_amount = account_balance * risk_per_trade
    price_diff = abs(entry_price - stop_loss_price)

    if price_diff == 0:
        raise ValueError("entry_price and stop_loss_price cannot be equal")

    return risk_amount / price_diff


def risk_reward_ratio(entry_price: float, stop_loss_price: float,
                       take_profit_price: float) -> float:
    """Calculate the risk/reward ratio of a trade."""
    risk = abs(entry_price - stop_loss_price)
    reward = abs(take_profit_price - entry_price)

    if risk == 0:
        raise ValueError("entry_price and stop_loss_price cannot be equal")

    return reward / risk


def max_drawdown(equity_curve) -> float:
    """
    Calculate the maximum drawdown of an equity curve.

    Parameters
    ----------
    equity_curve : list or pandas Series of portfolio values over time

    Returns
    -------
    Maximum drawdown as a negative fraction (e.g. -0.25 = -25%).
    """
    peak = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (value - peak) / peak
        if drawdown < max_dd:
            max_dd = drawdown

    return max_dd
