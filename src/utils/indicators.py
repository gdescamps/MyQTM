def compute_rsi(prices, period=14):
    """
    Calculates RSI (Relative Strength Index) using EMA smoothing.

    :param prices: List of prices (most recent first)
    :param period: RSI period, default 14
    :return: RSI value
    """
    if len(prices) != period + 1:
        return 50.0

    gains = []
    losses = []

    for i in range(0, period):
        delta = prices[i] - prices[i + 1]
        if delta > 0:
            gains.append(delta)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-delta)

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        return 100.0  # RSI maximum

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi
