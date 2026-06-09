"""
build_qualifying_dataset.py
============================
Builds a qualifying lap time dataset for Formula 1 seasons 2022-2025
using FastF1 as the sole data source.

The qualifying session ("Q") is loaded once per event and then split
into Q1, Q2, Q3 via the official FastF1 method:
    q1, q2, q3 = session.laps.split_qualifying_sessions()

Output: qualifying_dataset.csv
Each row = one driver's personal best lap in a given sub-session.

Requirements:
    pip install fastf1 pandas numpy

Usage:
    python build_qualifying_dataset.py
"""

import os
import warnings
import logging

import fastf1
import numpy as np
import pandas as pd

# ── Suppress verbose FastF1 logging ──────────────────────────────────────────
logging.getLogger("fastf1").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── Configuration ─────────────────────────────────────────────────────────────
SEASONS = [2022, 2023, 2024, 2025]
CACHE_DIR = "./f1_cache"
OUTPUT_FILE = "qualifying_dataset.csv"

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


# ── Helpers ───────────────────────────────────────────────────────────────────


def timedelta_to_seconds(td) -> float:
    """Convert a pandas Timedelta to total seconds; returns NaN if null."""
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
    """Return median weather values across the session."""
    cols = ["AirTemp", "TrackTemp", "Humidity", "Pressure", "WindSpeed", "Rainfall"]
    snapshot = {}
    for col in cols:
        try:
            if weather_df is not None and col in weather_df.columns:
                snapshot[col] = float(weather_df[col].median())
            else:
                snapshot[col] = np.nan
        except Exception:
            snapshot[col] = np.nan
    return snapshot


def get_best_lap_per_driver(laps: pd.DataFrame) -> pd.DataFrame:
    """
    Return each driver's personal best (minimum LapTime) from a
    sub-session laps DataFrame. Excludes laps with no recorded time.
    """
    if laps is None or laps.empty:
        return pd.DataFrame()
    valid = laps[laps["LapTime"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    idx = valid.groupby("Driver")["LapTime"].idxmin()
    return valid.loc[idx].copy()


def safe_get(row, key, default=np.nan):
    """Safely retrieve a value from a Series row."""
    try:
        val = row[key]
        if pd.isnull(val):
            return default
        return val
    except Exception:
        return default


def get_grid_position(driver: str, results_df) -> float:
    """Look up a driver's final grid position from session results."""
    try:
        if results_df is not None and driver in results_df["Abbreviation"].values:
            r = results_df[results_df["Abbreviation"] == driver].iloc[0]
            return safe_get(r, "Position", np.nan)
    except Exception:
        pass
    return np.nan


def build_row(
    lap_row,
    results_df,
    sub_session: str,
    year: int,
    round_number: int,
    circuit: str,
    weather: dict,
) -> dict:
    """Assemble one dataset row from a lap record and session metadata."""
    driver = safe_get(lap_row, "Driver", np.nan)
    team = safe_get(lap_row, "Team", np.nan)

    return {
        # Identifiers
        "Year": year,
        "Round": round_number,
        "Circuit": circuit,
        "Driver": driver,
        "Team": team,
        "SubSession": sub_session,  # Q1 / Q2 / Q3
        # Targets
        "LapTime_s": timedelta_to_seconds(safe_get(lap_row, "LapTime")),
        "GridPosition": get_grid_position(driver, results_df),
        # Lap features
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
        # Weather (median over the full qualifying session)
        "AirTemp_C": weather.get("AirTemp", np.nan),
        "TrackTemp_C": weather.get("TrackTemp", np.nan),
        "Humidity_pct": weather.get("Humidity", np.nan),
        "Pressure_hPa": weather.get("Pressure", np.nan),
        "WindSpeed_ms": weather.get("WindSpeed", np.nan),
        "Rainfall": weather.get("Rainfall", np.nan),
    }


# ── Main extraction loop ──────────────────────────────────────────────────────


def build_dataset(seasons: list) -> pd.DataFrame:
    all_rows = []

    for year in seasons:
        print(f"\n{'=' * 60}")
        print(f"  Season {year}")
        print(f"{'=' * 60}")

        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"  [ERROR] Could not load schedule for {year}: {e}")
            continue

        for _, event in schedule.iterrows():
            round_number = int(event["RoundNumber"])
            circuit = event.get(
                "OfficialEventName", event.get("EventName", f"Round{round_number}")
            )
            country = event.get("Country", "")
            print(f"\n  Round {round_number:02d} — {circuit} ({country})")

            # ── Load the single qualifying session ────────────────────────
            try:
                session = fastf1.get_session(year, round_number, "Q")
                # messages=True is required for split_qualifying_sessions()
                session.load(laps=True, weather=True, telemetry=False, messages=True)
            except Exception as e:
                print(f"    [SKIP] Could not load qualifying: {e}")
                continue

            # ── Weather snapshot (median over whole session) ───────────────
            weather = get_weather_snapshot(getattr(session, "weather_data", None))

            # ── Results for grid position lookup ──────────────────────────
            results_df = None
            try:
                results_df = session.results
            except Exception:
                pass

            # ── Split laps into Q1, Q2, Q3 ───────────────────────────────
            try:
                q1_laps, q2_laps, q3_laps = session.laps.split_qualifying_sessions()
            except Exception as e:
                print(f"    [SKIP] Could not split qualifying sessions: {e}")
                continue

            sub_session_map = {
                "Q1": q1_laps,
                "Q2": q2_laps,
                "Q3": q3_laps,
            }

            for sub_label, sub_laps in sub_session_map.items():
                best_laps = get_best_lap_per_driver(sub_laps)
                if best_laps.empty:
                    print(f"    [{sub_label}] No valid laps")
                    continue

                n = 0
                for _, lap_row in best_laps.iterrows():
                    row = build_row(
                        lap_row,
                        results_df,
                        sub_label,
                        year,
                        round_number,
                        circuit,
                        weather,
                    )
                    all_rows.append(row)
                    n += 1

                print(f"    [{sub_label}] {n} driver laps extracted")

    return pd.DataFrame(all_rows)


# ── Post-processing ───────────────────────────────────────────────────────────


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Standardise compound names
    if "Compound" in df.columns:
        df["Compound"] = df["Compound"].str.upper().str.strip()

    # Ordinal encoding for sub-session
    df["SubSession_ord"] = df["SubSession"].map({"Q1": 1, "Q2": 2, "Q3": 3})

    # Binary wet track flag
    if "Rainfall" in df.columns:
        df["Wet"] = (df["Rainfall"].fillna(0) > 0).astype(int)

    # Sort chronologically, then by lap time within each sub-session
    df = df.sort_values(["Year", "Round", "SubSession_ord", "LapTime_s"]).reset_index(
        drop=True
    )

    # Missing value report
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("\n── Missing values per column ──────────────────────")
        print(missing.to_string())

    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_raw = build_dataset(SEASONS)

    if df_raw.empty:
        print("\n[WARNING] No data extracted. Check internet connection.")
    else:
        df = postprocess(df_raw)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n{'=' * 60}")
        print(f"  Saved:    {OUTPUT_FILE}")
        print(f"  Shape:    {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"  Seasons:  {sorted(df['Year'].unique().tolist())}")
        print(f"  Circuits: {df['Circuit'].nunique()} unique events")
        print(f"  Drivers:  {df['Driver'].nunique()} unique drivers")
        print(f"{'=' * 60}\n")
        print(df.head(10).to_string(index=False))
