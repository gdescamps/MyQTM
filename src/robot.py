import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import src.config as config
from src.benchmark import build_trade_data
from src.data_pipeline import run_pipeline
from src.ib import (
    callback_cash_iteractive_broker,
    callback_close_positions_iteractive_broker,
    callback_ib_connected,
    callback_net_liquidation_iteractive_broker,
    callback_open_positions_iteractive_broker,
    callback_paris_open,
    callback_us_open,
)
from src.utils.ib import ib_reboot_docker
from src.utils.path import get_project_root
from src.utils.printlog import PrintLog
from src.utils.trade import (
    close_positions,
    get_param,
    open_positions,
    select_positions_to_close,
    select_positions_to_open,
)

assert config.TRAIN_DIR is not None


local_log = PrintLog(extra_name="_robot", log_time=True, enable=False)
capital = None
positions = None
positions_history = None
prev_date = None


# Daily download data
def daily_download_data():
    global local_log
    with local_log:
        print("daily_download_data():")

    # # wait until IB connected
    # while True:
    #     if not callback_ib_connected(local_log):
    #         with local_log:
    #             print("cannot connect to IB.")
    #     else:
    #         print("IB connected.")
    #         break
    #     time.sleep(1)

    # # check if Paris market is open
    # is_open = callback_paris_open(local_log)
    # if not is_open:
    #     with local_log:
    #         print("paris not open today, no trade.")
    #         print("done.")
    #     return

    # check if today is a weekday (Mon-Fri) in Europe/Paris
    paris_now = datetime.now(ZoneInfo("Europe/Paris"))
    if paris_now.weekday() >= 5:
        with local_log:
            print("today is weekend, no trade.")
            print("done.")
        return

    current_date = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d")
    config.TRADE_END_DATE = current_date

    run_pipeline(config, log_local=local_log)
    # TODO check if data consistency after run_pipeline

    with local_log:
        print("done.")


# Daily trade positions
def daily_trade_positions():
    global local_log, capital, positions, positions_history, prev_date
    global positions_long_to_open, positions_long_to_close, positions_short_to_open, positions_short_to_close

    positions_long_to_open = []
    positions_long_to_close = []
    positions_short_to_open = []
    positions_short_to_close = []

    with local_log:
        print("daily_trade_positions():")

    # wait until IB connected
    while True:
        if not callback_ib_connected(log=local_log):
            with local_log:
                print("cannot connect to IB.")
                print("rebooting docker.")
            ib_reboot_docker()
        else:
            print("IB connected.")
            break
        time.sleep(1)

    is_open = callback_paris_open(log=local_log)
    if not is_open:
        with local_log:
            print("paris not open today, no trade.")
            print("done.")
        return

    current_date = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d")
    config.TRADE_END_DATE = current_date

    with local_log:
        print(f"current_date: {current_date}")

    with open(os.path.join(config.TRAIN_DIR, "best_params.json"), "r") as f:
        XBEST = json.load(f)
    (
        # max_positions,
        long_open_prob_thresa,
        long_close_prob_thresa,
        short_open_prob_thresa,
        short_close_prob_thresa,
        long_open_prob_thresb,
        long_close_prob_thresb,
        short_open_prob_thresb,
        short_close_prob_thresb,
        long_prob_powera,
        short_prob_powera,
        long_prob_powerb,
        short_prob_powerb,
    ) = list(XBEST)

    bench_start_date = pd.to_datetime(config.TEST_START_DATE, format="%Y-%m-%d")
    bench_end_date = pd.to_datetime(current_date, format="%Y-%m-%d")
    data_path = Path(get_project_root()) / "data" / "fmp_data"
    ib_path = Path(get_project_root()) / "data" / "ib_data"
    model_path = Path(get_project_root()) / config.TRAIN_DIR

    if positions is None:
        if not os.path.exists(ib_path / "positions.json"):
            positions = []
        else:
            with open(ib_path / "positions.json", "r") as f:
                positions = json.load(f)

    if positions_history is None:
        if not os.path.exists(ib_path / "positions_history.json"):
            positions_history = []
        else:
            with open(ib_path / "positions_history.json", "r") as f:
                positions_history = json.load(f)

    with open(ib_path / f"{current_date}_positions_before.json", "w") as f:
        json.dump(positions, f, indent=2)

    with open(ib_path / f"{current_date}_positions_history_before.json", "w") as f:
        json.dump(positions_history, f, indent=2)

    trade_data = build_trade_data(
        model_path=model_path,
        data_path=data_path,
        file_date_str=TODO,
        start_date=bench_start_date,
        end_date=bench_end_date,
        end_limit=False,  # do not limit to TEST_END_DATE
    )

    yesterday_date = list(sorted(trade_data.keys()))[-1]
    prev_date = list(sorted(trade_data.keys()))[-2]
    prev_item = trade_data[prev_date]
    prev_item = prev_item.copy()

    capital = callback_cash_iteractive_broker(log=local_log)
    capital_and_position = callback_net_liquidation_iteractive_broker(log=local_log)
    position_size = capital_and_position / config.MAX_POSITIONS

    last_date_processed = None
    if os.path.exists(ib_path / "last_date_processed.json"):
        with open(ib_path / "last_date_processed.json", "r") as f:
            last_date_processed = json.load(f)

    if last_date_processed is not None and pd.to_datetime(
        yesterday_date
    ) <= pd.to_datetime(last_date_processed):
        with local_log:
            print(
                f"yesterday_date {yesterday_date} <= last_date_processed {last_date_processed}, no trade."
            )
        return

    (
        long_open_prob_thres,
        short_open_prob_thres,
        long_close_prob_thres,
        short_close_prob_thres,
        long_prob_power,
        short_prob_power,
    ) = get_param(
        yesterday_date,
        long_open_prob_thresa,
        long_close_prob_thresa,
        short_open_prob_thresa,
        short_close_prob_thresa,
        long_open_prob_thresb,
        long_close_prob_thresb,
        short_open_prob_thresb,
        short_close_prob_thresb,
        long_prob_powera,
        short_prob_powera,
        long_prob_powerb,
        short_prob_powerb,
        end_limit=False,
    )

    # compute sorted candidates items
    item = trade_data[yesterday_date]
    long_item = item.copy()
    long_item = dict(
        sorted(long_item.items(), key=lambda x: x[1]["ybull"], reverse=True)
    )

    short_item = item.copy()
    short_item = dict(
        sorted(short_item.items(), key=lambda x: x[1]["ybear"], reverse=True)
    )

    stock_filter = config.TRADE_STOCKS.copy()

    # compute long position to open
    new_open_ybull = 0
    positions_long_to_open, new_open_ybull = select_positions_to_open(
        long_item,
        prev_item,
        positions,
        positions_long_to_open,
        stock_filter,
        config.MAX_POSITIONS,
        class_val=2,
        prob_key="ybull",
        open_prob_thres=long_open_prob_thres,
        new_open_yprob=new_open_ybull,
    )
    with open(ib_path / f"{current_date}_positions_long_to_open.json", "w") as f:
        json.dump(positions_long_to_open, f, indent=2)

    # compute short position to open
    new_open_ybear = 0
    positions_short_to_open, new_open_ybear = select_positions_to_open(
        short_item,
        prev_item,
        positions,
        positions_short_to_open,
        stock_filter,
        config.MAX_POSITIONS,
        class_val=0,
        prob_key="ybear",
        open_prob_thres=short_open_prob_thres,
        new_open_yprob=new_open_ybear,
    )

    with open(ib_path / f"{current_date}_positions_short_to_open.json", "w") as f:
        json.dump(positions_short_to_open, f, indent=2)

    # compute positions to close
    positions_long_to_close, positions_short_to_close, remove_pos_indexes = (
        select_positions_to_close(
            positions,
            item,
            long_close_prob_thres,
            short_close_prob_thres,
            new_open_ybull,
            new_open_ybear,
            capital,
        )
    )

    with open(ib_path / f"{current_date}_positions_short_to_close.json", "w") as f:
        json.dump(positions_short_to_close, f, indent=2)

    with open(ib_path / f"{current_date}_positions_long_to_close.json", "w") as f:
        json.dump(positions_long_to_close, f, indent=2)

    wait_count = 0
    while True:
        try:
            is_open = callback_us_open(log=local_log)
            if is_open:
                break
            time.sleep(10)
            if wait_count % 10 == 0:
                with local_log:
                    print("us market not open, check every 10s...")
            wait_count += 1
            if wait_count > 400:
                with local_log:
                    print("us still not market not open, stop waiting.")
                return
        except Exception as e:
            print(f"Error disconnecting: {e}")
            pass

    # closes positions
    (
        capital,
        positions_history,
        positions_long_to_close,
        positions_short_to_close,
    ) = close_positions(
        positions_long_to_close,
        positions_short_to_close,
        trade_data,
        yesterday_date,
        capital,
        positions_history,
        callback_close_positions_iteractive_broker,
        log=local_log,
    )

    capital = callback_cash_iteractive_broker(log=local_log)
    capital_and_position = callback_net_liquidation_iteractive_broker(log=local_log)
    position_size = capital_and_position / config.MAX_POSITIONS

    # open new positions (long)
    positions, capital, positions_long_to_open = open_positions(
        positions_long_to_open,
        positions,
        trade_data,
        yesterday_date,
        capital,
        capital_and_position,
        position_size,
        config.MAX_POSITIONS,
        long_open_prob_thres,
        "long",
        long_prob_power,
        callback_open_positions_iteractive_broker,
        log=local_log,
    )

    # open new positions (short)
    positions, capital, positions_short_to_open = open_positions(
        positions_short_to_open,
        positions,
        trade_data,
        yesterday_date,
        capital,
        capital_and_position,
        position_size,
        config.MAX_POSITIONS,
        short_open_prob_thres,
        "short",
        short_prob_power,
        callback_open_positions_iteractive_broker,
        log=local_log,
    )

    positions = [pos for i, pos in enumerate(positions) if i not in remove_pos_indexes]

    with open(ib_path / f"{current_date}_positions_after.json", "w") as f:
        json.dump(positions, f, indent=2)

    with open(ib_path / f"{current_date}_positions_history_after.json", "w") as f:
        json.dump(positions_history, f, indent=2)

    try:
        if positions is not None:
            with open(ib_path / "positions.json", "w") as f:
                json.dump(positions, f, indent=2)

        if positions_history is not None:
            with open(ib_path / "positions_history.json", "w") as f:
                json.dump(positions_history, f, indent=2)

        with open(ib_path / "last_date_processed.json", "w") as f:
            json.dump(yesterday_date, f, indent=2)

    except KeyboardInterrupt:
        with local_log:
            print("Interruption detected, file saving completed.")

    with local_log:
        print("done.")


# Main loop
def main():
    global local_log

    # comment for direct debug purposes
    # try:
    #     # daily_download_data()
    #     daily_trade_positions()
    # except KeyboardInterrupt:
    #     # clean exit on user interruption
    #     with local_log:
    #         print("Interruption detected, stopping the robot.")
    # except Exception as e:
    #     with local_log:
    #         print("Crash in main loop:", e)

    paris_tz = ZoneInfo("Europe/Paris")

    def next_run_time(hour, minute):
        now = datetime.now(paris_tz)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now >= target:
            target += timedelta(days=1)
        return target

    while True:
        try:
            next_download = next_run_time(13, 0)
            next_trade = next_run_time(15, 30)
            next_run_dt = min(next_download, next_trade)
            sleep_seconds = (next_run_dt - datetime.now(paris_tz)).total_seconds()

            time.sleep(sleep_seconds)
            now = datetime.now(paris_tz)
            if abs((now - next_download).total_seconds()) < 120:
                daily_download_data()

            if abs((now - next_trade).total_seconds()) < 120:
                daily_trade_positions()

        except KeyboardInterrupt:
            # clean exit on user interruption
            with local_log:
                print("Interruption detected, stopping the robot.")
            break
        except Exception as e:
            with local_log:
                print(f"Crash in main loop: {e}")
            time.sleep(1)
            pass


if __name__ == "__main__":
    main()
