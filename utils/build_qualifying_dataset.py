"""
build_qualifying_dataset.py
============================
Builds a qualifying lap time dataset for F1 seasons 2022-2025 using
FastF1 as the sole data source.

Two output files are produced:
    qualifying_dataset.csv          — qualifying features only
    qualifying_dataset_with_fp.csv  — same rows + FP1/FP2/FP3 best laps

RATE LIMIT PROTECTION
---------------------
The FastF1 API enforces a limit of 500 calls/hour. This script:
    - Sleeps SLEEP_BETWEEN_SESSIONS seconds between every session load.
    - Uses a checkpoint file (checkpoint.json) to skip (year, round) pairs
    that were already successfully extracted in a previous run.
    - On RateLimitExceededError, waits SLEEP_ON_RATELIMIT seconds and
    retries once before giving up on that session.

Resume after interruption:
    Simply re-run the script. Already-extracted rounds are skipped and
    the new rows are appended to the existing CSV files.

Requirements:
    pip install fastf1 pandas numpy

Usage:
    python build_qualifying_dataset.py
"""

import json
import os
import time
import warnings
import logging

import fastf1
import fastf1.exceptions
import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.getLogger("fastf1").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
SEASONS = [2022, 2023, 2024, 2025]
CACHE_DIR = "./f1_cache"
OUTPUT_BASE = "data/qualifying_dataset.csv"
OUTPUT_FP = "data/qualifying_dataset_with_fp.csv"
CHECKPOINT_FILE = "data/checkpoint.json"

# Seconds to wait between each session.load() call.
# 4 sessions per round (Q + FP1 + FP2 + FP3) × ~24 rounds × 4 seasons
# = ~384 loads. At 8s each = ~51 min total, well within the hourly limit.
SLEEP_BETWEEN_SESSIONS = 8

# How long to wait (seconds) when a rate limit error is hit before retrying.
SLEEP_ON_RATELIMIT = 120

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


# ── Checkpoint helpers ────────────────────────────────────────────────────────


def load_checkpoint() -> set:
    """Return a set of (year, round) tuples already processed."""
    if not os.path.exists(CHECKPOINT_FILE):
        return set()
    with open(CHECKPOINT_FILE) as f:
        data = json.load(f)
    return {tuple(x) for x in data}


def save_checkpoint(done: set) -> None:
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump([list(x) for x in done], f)


# ── Session loader with rate-limit retry ─────────────────────────────────────


def load_session_safe(
    year, round_number, identifier, laps=True, weather=False, messages=False
) -> object | None:
    """
    Load a FastF1 session with a sleep before the call and one retry
    on RateLimitExceededError. Returns None on failure.
    """
    time.sleep(SLEEP_BETWEEN_SESSIONS)
    for attempt in range(2):
        try:
            sess = fastf1.get_session(year, round_number, identifier)
            sess.load(laps=laps, weather=weather, telemetry=False, messages=messages)
            return sess
        except fastf1.exceptions.RateLimitExceededError:
            if attempt == 0:
                print(
                    f"      [RATE LIMIT] Waiting {SLEEP_ON_RATELIMIT}s before retry..."
                )
                time.sleep(SLEEP_ON_RATELIMIT)
            else:
                print(
                    f"      [RATE LIMIT] Giving up on "
                    f"{year} R{round_number} {identifier}"
                )
                return None
        except Exception as e:
            print(f"      [SKIP {identifier}] {e}")
            return None
    return None


# ── Data helpers ──────────────────────────────────────────────────────────────


def timedelta_to_seconds(td) -> float:
    try:
        if pd.isnull(td):
            return np.nan
    except Exception:
        pass
    try:
        return td.total_seconds()
    except Exception:
        return np.nan


def get_weather_snapshot(weather_df) -> dict:
    cols = ["AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall"]
    out = {}
    for col in cols:
        try:
            out[col] = (
                float(weather_df[col].median())
                if weather_df is not None and col in weather_df.columns
                else np.nan
            )
        except Exception:
            out[col] = np.nan
    return out


def get_best_lap_per_driver(laps) -> pd.DataFrame:
    if laps is None or (hasattr(laps, "empty") and laps.empty):
        return pd.DataFrame()
    valid = laps[laps["LapTime"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    return valid.loc[valid.groupby("Driver")["LapTime"].idxmin()].copy()


def safe_get(row, key, default=np.nan):
    try:
        val = row[key]
        return default if pd.isnull(val) else val
    except Exception:
        return default


def get_grid_position(driver: str, results_df) -> float:
    try:
        if results_df is not None and driver in results_df["Abbreviation"].values:
            r = results_df[results_df["Abbreviation"] == driver].iloc[0]
            return safe_get(r, "Position", np.nan)
    except Exception:
        pass
    return np.nan


def is_sprint_weekend(event) -> bool:
    for i in range(1, 6):
        val = event.get(f"Session{i}", "")
        if isinstance(val, str) and "sprint" in val.lower():
            return True
    return False


def load_fp_best_times(year: int, round_number: int, sprint: bool) -> dict:
    """
    Load FP1, FP2, FP3 best lap times per driver.
    On sprint weekends, FP3 is replaced by SQ or SS where available.
    Returns dict: driver -> {FP1_s, FP2_s, FP3_s}
    """
    sessions_to_load = {"FP1": "FP1", "FP2": "FP2"}
    sessions_to_load["FP3"] = None if sprint else "FP3"

    results: dict = {}

    for label, identifier in sessions_to_load.items():
        col = f"{label}_s"

        if identifier is None:
            # Sprint weekend: try SQ then SS
            sess = None
            for sid in ("SQ", "SS"):
                sess = load_session_safe(year, round_number, sid)
                if sess is not None:
                    break
        else:
            sess = load_session_safe(year, round_number, identifier)

        if sess is None:
            continue

        best = get_best_lap_per_driver(sess.laps)
        if best.empty:
            continue

        for _, row in best.iterrows():
            drv = safe_get(row, "Driver", None)
            if drv is None:
                continue
            if drv not in results:
                results[drv] = {"FP1_s": np.nan, "FP2_s": np.nan, "FP3_s": np.nan}
            results[drv][col] = timedelta_to_seconds(safe_get(row, "LapTime"))

    return results


def build_row(
    lap_row,
    results_df,
    sub_session: str,
    year: int,
    round_number: int,
    circuit: str,
    weather: dict,
) -> dict:
    driver = safe_get(lap_row, "Driver", np.nan)
    team = safe_get(lap_row, "Team", np.nan)
    return {
        "Year": year,
        "Round": round_number,
        "Circuit": circuit,
        "Driver": driver,
        "Team": team,
        "SubSession": sub_session,
        "LapTime_s": timedelta_to_seconds(safe_get(lap_row, "LapTime")),
        "GridPosition": get_grid_position(driver, results_df),
        "Sector1_s": timedelta_to_seconds(safe_get(lap_row, "Sector1Time")),
        "Sector2_s": timedelta_to_seconds(safe_get(lap_row, "Sector2Time")),
        "Sector3_s": timedelta_to_seconds(safe_get(lap_row, "Sector3Time")),
        "TyreLife": safe_get(lap_row, "TyreLife"),
        "Compound": safe_get(lap_row, "Compound"),
        "FreshTyre": safe_get(lap_row, "FreshTyre"),
        "SpeedI1": safe_get(lap_row, "SpeedI1"),
        "SpeedI2": safe_get(lap_row, "SpeedI2"),
        "SpeedFL": safe_get(lap_row, "SpeedFL"),
        "SpeedST": safe_get(lap_row, "SpeedST"),
        "AirTemp_C": weather.get("AirTemp", np.nan),
        "TrackTemp_C": weather.get("TrackTemp", np.nan),
        "Humidity_pct": weather.get("Humidity", np.nan),
        "Pressure_hPa": weather.get("Pressure", np.nan),
        "WindSpeed_ms": weather.get("WindSpeed", np.nan),
        "Rainfall": weather.get("Rainfall", np.nan),
    }


# ── Main extraction loop ──────────────────────────────────────────────────────


def build_dataset(
    seasons: list, existing_base: pd.DataFrame, existing_fp: pd.DataFrame, done: set
) -> tuple[list, list]:
    """
    Returns two lists of new rows (base and fp) not yet in `done`.
    """
    new_base = []
    new_fp = []

    for year in seasons:
        print(f"\n{'=' * 60}")
        print(f"  Season {year}")
        print(f"{'=' * 60}")

        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"  [ERROR] Schedule: {e}")
            continue

        for _, event in schedule.iterrows():
            round_number = int(event["RoundNumber"])
            key = (year, round_number)

            if key in done:
                print(f"\n  Round {round_number:02d} — [already done, skipping]")
                continue

            circuit = event.get(
                "OfficialEventName", event.get("EventName", f"Round{round_number}")
            )
            country = event.get("Country", "")
            sprint = is_sprint_weekend(event)

            print(
                f"\n  Round {round_number:02d} — {circuit} ({country})"
                f"{'  [SPRINT]' if sprint else ''}"
            )

            # ── Qualifying ────────────────────────────────────────────────
            q_sess = load_session_safe(
                year, round_number, "Q", laps=True, weather=True, messages=True
            )
            if q_sess is None:
                continue

            weather = get_weather_snapshot(getattr(q_sess, "weather_data", None))
            results_df = None
            try:
                results_df = q_sess.results
            except Exception:
                pass

            try:
                q1_laps, q2_laps, q3_laps = q_sess.laps.split_qualifying_sessions()
            except Exception as e:
                print(f"    [SKIP split] {e}")
                continue

            # ── FP times ──────────────────────────────────────────────────
            print(f"    Loading FP sessions...", end=" ", flush=True)
            fp_times = load_fp_best_times(year, round_number, sprint)
            n_fp_drivers = sum(
                1 for v in fp_times.values() if any(not np.isnan(x) for x in v.values())
            )
            print(f"{n_fp_drivers} drivers with FP data")

            # ── Build rows ────────────────────────────────────────────────
            round_had_data = False
            for sub_label, sub_laps in [
                ("Q1", q1_laps),
                ("Q2", q2_laps),
                ("Q3", q3_laps),
            ]:
                best = get_best_lap_per_driver(sub_laps)
                if best.empty:
                    print(f"    [{sub_label}] No valid laps")
                    continue

                n = 0
                for _, lap_row in best.iterrows():
                    row = build_row(
                        lap_row,
                        results_df,
                        sub_label,
                        year,
                        round_number,
                        circuit,
                        weather,
                    )
                    new_base.append(row)

                    drv_fp = fp_times.get(row["Driver"], {})
                    new_fp.append(
                        {
                            **row,
                            "FP1_s": drv_fp.get("FP1_s", np.nan),
                            "FP2_s": drv_fp.get("FP2_s", np.nan),
                            "FP3_s": drv_fp.get("FP3_s", np.nan),
                            "IsSprintWeekend": int(sprint),
                        }
                    )
                    n += 1
                    round_had_data = True

                print(f"    [{sub_label}] {n} driver laps extracted")

            if round_had_data:
                done.add(key)
                save_checkpoint(done)

    return new_base, new_fp


# ── Post-processing ───────────────────────────────────────────────────────────


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].str.upper().str.strip()
    df["SubSession_ord"] = df["SubSession"].map({"Q1": 1, "Q2": 2, "Q3": 3})
    if "Rainfall" in df.columns:
        df["Wet"] = (df["Rainfall"].fillna(0) > 0).astype(int)
    df = df.sort_values(["Year", "Round", "SubSession_ord", "LapTime_s"]).reset_index(
        drop=True
    )
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load existing data and checkpoint
    existing_base = (
        pd.read_csv(OUTPUT_BASE) if os.path.exists(OUTPUT_BASE) else pd.DataFrame()
    )
    existing_fp = (
        pd.read_csv(OUTPUT_FP) if os.path.exists(OUTPUT_FP) else pd.DataFrame()
    )
    done = load_checkpoint()

    # Pre-populate checkpoint from existing base CSV if checkpoint is empty
    # (handles the case where the base CSV was produced by the old script)
    if not done and not existing_base.empty:
        for (y, r), _ in existing_base.groupby(["Year", "Round"]):
            done.add((int(y), int(r)))
        save_checkpoint(done)
        print(
            f"Checkpoint initialised from existing CSV: "
            f"{len(done)} rounds already done."
        )

    new_base_rows, new_fp_rows = build_dataset(
        SEASONS, existing_base, existing_fp, done
    )

    if not new_base_rows:
        print("\nNo new data extracted.")
    else:
        new_base = postprocess(pd.DataFrame(new_base_rows))
        new_fp = postprocess(pd.DataFrame(new_fp_rows))

        # Append to existing data
        df_base = postprocess(pd.concat([existing_base, new_base], ignore_index=True))
        df_fp = postprocess(pd.concat([existing_fp, new_fp], ignore_index=True))

        df_base.to_csv(OUTPUT_BASE, index=False)
        df_fp.to_csv(OUTPUT_FP, index=False)

        for label, df, path in [
            ("Base (qualifying only)", df_base, OUTPUT_BASE),
            ("With FP times", df_fp, OUTPUT_FP),
        ]:
            print(f"\n{'=' * 60}")
            print(f"  {label}")
            print(f"  Saved:    {path}")
            print(f"  Shape:    {df.shape[0]} rows x {df.shape[1]} columns")
            print(f"  Seasons:  {sorted(df['Year'].unique().tolist())}")
            print(f"  Circuits: {df['Circuit'].nunique()} unique events")
            print(f"  Drivers:  {df['Driver'].nunique()} unique drivers")
            print(f"{'=' * 60}")

        missing = df_base.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            print("\n── Missing values (base) ───────────────────────────")
            print(missing.to_string())
