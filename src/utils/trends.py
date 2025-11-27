import numpy as np
import pandas as pd


def detect_trends(
    df: pd.DataFrame,
    ma_short: int = 4,
    ma_mid: int = 8,
    ma_long: int = 30,
    min_days: int = 8,
    avg_daily_percent_thr: float = 0.15,  # 0.15% per day
) -> list[dict]:
    """
    Detect bullish, bearish, and range trends on a *daily* OHLCV DataFrame.
    Bullish and bearish segments are kept only if their average daily return exceeds
    `avg_daily_percent_thr` in absolute value and in the right direction. Any plateau
    between mid-vs-long EMAs lasting at least `min_days` that does not qualify as
    bullish or bearish is labeled as "range".

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a DatetimeIndex and a 'close' column.
    ma_short, ma_mid, ma_long : int
        Windows (in trading days) for the exponential moving averages.
    min_days : int
        Minimum segment duration (in days) defined by the mid-vs-long EMA plateau.
    avg_daily_percent_thr : float
        Minimum average daily return required to keep a trend:
        +thr for bullish trends, −thr for bearish. Expressed as a percentage
        (0.15 == 0.15% per day).

    Returns
    -------
    list[dict]
        Each dict contains { 'trend', 'start', 'end', 'avg_daily_return', 'total_return', 'days' }.
    """
    df = df.sort_index().copy()

    # 1️⃣  EMAs
    df["ema_short"] = df["close"].ewm(span=ma_short, adjust=False).mean()
    df["ema_mid"] = df["close"].ewm(span=ma_mid, adjust=False).mean()
    df["ema_long"] = df["close"].ewm(span=ma_long, adjust=False).mean()

    # 2️⃣  Mid-vs-Long crossovers
    sign_ml = np.sign(df["ema_mid"] - df["ema_long"]).replace(0, np.nan).ffill()
    crosses_ml = sign_ml.ne(sign_ml.shift())
    cross_ml_idx = df.index[crosses_ml]

    # 3️⃣  Short-vs-Mid crossovers
    sign_sm = np.sign(df["ema_short"] - df["ema_mid"]).replace(0, np.nan).ffill()
    crosses_sm = sign_sm.ne(sign_sm.shift())
    cross_sm_idx = df.index[crosses_sm]

    trends = []
    last_end = None  # Pour éviter les chevauchements

    # 4️⃣  Build raw segments
    for i in range(1, len(cross_ml_idx)):
        start_ml, end_ml = cross_ml_idx[i - 1], cross_ml_idx[i]
        # Skip segments shorter than minimum duration
        if (end_ml - start_ml).days < min_days:
            continue

        # nearest short-mid cross *before* each mid-long cross
        if any(cross_sm_idx < start_ml):
            adj_start = cross_sm_idx[cross_sm_idx < start_ml][-1]
        else:
            adj_start = start_ml

        if any(cross_sm_idx < end_ml):
            adj_end = cross_sm_idx[cross_sm_idx < end_ml][-1]
        else:
            adj_end = end_ml

        # Empêcher les chevauchements
        if last_end is not None and adj_start <= last_end:
            position = df.index.get_loc(last_end)
            if position + 1 >= len(df.index):
                continue
            adj_start = df.index[position + 1]
            if adj_start > adj_end:
                continue  # segment invalide

        # compute average daily return over the segment
        days = int((adj_end - adj_start).days or 1)  # avoid division by zero
        ret = float(df.loc[adj_end, "close"] / df.loc[adj_start, "close"] - 1)
        ret_per_days = float(ret / days)

        # Determine potential trend type from mid-long slope at start of segment
        trend_type_ml = "bullish" if sign_ml.loc[start_ml] > 0 else "bearish"

        # 5️⃣  filter on performance direction & magnitude; otherwise mark as range
        if trend_type_ml == "bullish" and ret_per_days >= avg_daily_percent_thr / 100:
            trends.append(
                {
                    "trend": "bullish",
                    "start": str(adj_start.date()),
                    "end": str(adj_end.date()),
                    "total_return": 100 * ret,
                    "days": days,
                }
            )
            last_end = adj_end  # Mettre à jour la dernière fin
        elif (
            trend_type_ml == "bearish" and ret_per_days <= -avg_daily_percent_thr / 100
        ):
            trends.append(
                {
                    "trend": "bearish",
                    "start": str(adj_start.date()),
                    "end": str(adj_end.date()),
                    "total_return": 100 * ret,
                    "days": days,
                }
            )
            last_end = adj_end  # Mettre à jour la dernière fin

    return trends


def score_from_df(df: pd.DataFrame, date: str, decay: float = 0.9) -> int:
    """
    Calcule la classe de gain/perte en pourcentage entre open d+1 et open d+6.

    - df : DataFrame avec index = date (datetime), colonnes : 'open'
    - date : point de référence (str, ex : '2025-05-27')
    - decay : ignoré (pour compatibilité)

    Retourne un entier entre 0 et 6 selon les classes définies.
    """
    df = df.sort_index()
    if date not in df.index:
        return None, None

    start_idx = df.index.get_loc(date) + 1
    end_idx = (
        start_idx + 5
    )  # d+1 à d+6 inclus = 6 valeurs, donc d+1 à d+6 => [start_idx, end_idx]
    if end_idx >= len(df):
        return None, None

    open_d1 = df.iloc[start_idx]["open"]
    open_d6 = df.iloc[end_idx]["open"]
    if open_d1 == 0:
        return None, None

    pct = 100 * (open_d6 - open_d1) / open_d1

    # Classes :
    # 0: < -3.5%
    # 1: [-3.5%, -2%)
    # 2: [-2%, -0.5%)
    # 3: [-0.5%, 0.5%)
    # 4: [0.5%, 2%)
    # 5: [2%, 3.5%)
    # 6: >= 3.5%
    if pct < -3.5:
        classe = 0
    elif -3.5 <= pct < -2:
        classe = 1
    elif -2 <= pct < -0.5:
        classe = 2
    elif -0.5 <= pct < 0.5:
        classe = 3
    elif 0.5 <= pct < 2:
        classe = 4
    elif 2 <= pct < 3.5:
        classe = 5
    elif pct >= 3.5:
        classe = 6
    else:
        return None, None  # Cas improbable

    return classe, pct
