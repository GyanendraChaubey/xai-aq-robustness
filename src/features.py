"""
Feature engineering + AQI-category labeling for the Beijing (primary/search +
spatial-transfer) and Delhi (cross-city transfer) datasets.

Design decisions (documented here, referenced in the paper's Methods section):

1. DAILY resolution, not hourly. Both cities are aggregated to daily means
   (max for PM2.5 peak feature). This (a) matches the CPCB/NAQI 24-hr-average
   convention the target is defined on, (b) keeps the search-phase row count
   tractable for a 2-core sandbox, (c) makes Beijing and Delhi directly
   comparable (Delhi's CPCB station only has daily granularity).

2. Target = NEXT-DAY AQI category from PM2.5 (India CPCB breakpoints, 24-hr
   avg, µg/m3): Good 0-30, Satisfactory 31-60, Moderate 61-90, Poor 91-120,
   Very Poor 121-250, Severe >250. Predictors are day-t features; target is
   day-(t+1) category => no leakage (day-t PM2.5 is a legitimate
   autoregressive predictor of day-(t+1) PM2.5, not the target itself).

3. Station identity is DELIBERATELY EXCLUDED as a feature. Keeping it would
   let models use a per-station lookup shortcut that cannot transfer to an
   unseen station (no matching one-hot column) or to Delhi at all. Excluding
   it is what makes the leave-one-station-out and cross-city transfer tests
   meaningful rather than vacuous.

4. Common feature schema across cities (for cross-city transfer):
   pm25, pm10, so2, no2, co, o3, temp, pres, rh, rain, wspm, wd_sin, wd_cos
   + rolling 3-day / 7-day means of pm25 and temp + calendar (month, dow,
   cyclic-encoded).
   - Beijing has RH computed from TEMP+DEWP via Magnus formula (no direct RH
     column in the UCI data).
   - Beijing CO is in the same µg/m3 scale as PM/SO2/NO2; Delhi CO is
     reported in mg/m3 and is multiplied by 1000 for scale parity.
   - Delhi WSPM (wind speed) is unavailable in the usable 2021-2023 window
     (100% missing at source) -> left as NaN, handled by the pipeline's
     median imputer at inference. This is treated as a genuine cross-domain
     missing-sensor-channel case, not an error.
   - Wind direction encoded as sin/cos of degrees (Beijing's 16-point
     compass string is mapped to degrees first) so 350 deg and 10 deg score
     as adjacent, not distant.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/home/claude/xai_aq/data")

AQI_BREAKPOINTS = [  # (upper_bound_inclusive, label, code)
    (30, "Good", 0),
    (60, "Satisfactory", 1),
    (90, "Moderate", 2),
    (120, "Poor", 3),
    (250, "Very Poor", 4),
    (np.inf, "Severe", 5),
]
AQI_LABELS = [b[1] for b in AQI_BREAKPOINTS]

COMPASS_TO_DEG = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

COMMON_FEATURES = [
    "pm25", "pm10", "so2", "no2", "co", "o3",
    "temp", "pres", "rh", "rain", "wspm", "wd_sin", "wd_cos",
    "pm25_roll3", "pm25_roll7", "temp_roll3",
    "month_sin", "month_cos", "dow_sin", "dow_cos",
]

# Feature groups used for GROUPED SHAP (correlated meteorological drivers
# credited as a block rather than individually attributed).
FEATURE_GROUPS = {
    "pollutants": ["pm25", "pm10", "so2", "no2", "co", "o3"],
    "pollutants_roll": ["pm25_roll3", "pm25_roll7"],
    "meteorology": ["temp", "pres", "rh", "rain", "wspm", "wd_sin", "wd_cos", "temp_roll3"],
    "calendar": ["month_sin", "month_cos", "dow_sin", "dow_cos"],
}


def aqi_category(pm25):
    pm25 = np.asarray(pm25, dtype=float)
    codes = np.full(pm25.shape, np.nan)
    lower = -np.inf
    for upper, label, code in AQI_BREAKPOINTS:
        mask = (pm25 > lower) & (pm25 <= upper)
        codes[mask] = code
        lower = upper
    return codes


def _cyclic(series, period):
    rad = 2 * np.pi * series / period
    return np.sin(rad), np.cos(rad)


def _rh_from_temp_dewp(temp_c, dewp_c):
    """Magnus-formula relative humidity (%) from temperature and dewpoint, both C."""
    a, b = 17.625, 243.04
    def sat_vp(t):
        return np.exp((a * t) / (b + t))
    rh = 100 * sat_vp(dewp_c) / sat_vp(temp_c)
    return np.clip(rh, 0, 100)


def load_beijing_daily():
    df = pd.read_csv(DATA_DIR / "beijing_air_quality_cleaned.csv")
    df["date"] = pd.to_datetime(df["date"])
    df["wd_deg"] = df["wd"].map(COMPASS_TO_DEG)

    agg = {
        "PM2.5": "mean", "PM10": "mean", "SO2": "mean", "NO2": "mean",
        "CO": "mean", "O3": "mean", "TEMP": "mean", "PRES": "mean",
        "DEWP": "mean", "RAIN": "sum", "WSPM": "mean", "wd_deg": "mean",
    }
    daily = df.groupby(["station", "date"]).agg(agg).reset_index()
    daily = daily.rename(columns={
        "PM2.5": "pm25", "PM10": "pm10", "SO2": "so2", "NO2": "no2",
        "CO": "co", "O3": "o3", "TEMP": "temp", "PRES": "pres",
        "DEWP": "dewp", "RAIN": "rain", "WSPM": "wspm",
    })
    daily["rh"] = _rh_from_temp_dewp(daily["temp"], daily["dewp"])
    daily["wd_sin"], daily["wd_cos"] = _cyclic(daily["wd_deg"].fillna(0), 360)
    daily["city"] = "beijing"
    return daily


def load_delhi_daily():
    import glob
    frames = []
    for f in sorted(glob.glob(str(DATA_DIR / "delhi_src/data/raw/air_quality_*.csv"))):
        year = int(f.split("_")[-1].split(".")[0])
        if year < 2021:  # 2017/2018 are >85% missing at source -- excluded
            continue
        d = pd.read_csv(f)
        frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = pd.to_datetime(raw["Timestamp"], format="mixed", dayfirst=True)

    ren = {
        "PM2.5 (µg/m³)": "pm25", "PM10 (µg/m³)": "pm10", "SO2 (µg/m³)": "so2",
        "NO2 (µg/m³)": "no2", "CO (mg/m³)": "co_mg", "Ozone (µg/m³)": "o3",
        "AT (°C)": "temp", "RH (%)": "rh", "BP (mmHg)": "pres",
        "RF (mm)": "rain", "WD (deg)": "wd_deg",
    }
    raw = raw.rename(columns=ren)
    raw["co"] = raw["co_mg"] * 1000.0  # mg/m3 -> ug/m3, matches Beijing scale
    raw["wspm"] = np.nan  # not available at source for this station/window
    raw["wd_sin"], raw["wd_cos"] = _cyclic(raw["wd_deg"].fillna(0), 360)
    raw["station"] = "delhi_ncr"
    raw["city"] = "delhi"

    keep = ["city", "station", "date", "pm25", "pm10", "so2", "no2", "co", "o3",
            "temp", "pres", "rh", "rain", "wspm", "wd_sin", "wd_cos"]
    return raw[keep].sort_values(["station", "date"]).reset_index(drop=True)


def add_features_and_target(daily):
    """Adds rolling/calendar features and the next-day AQI-category target.
    Operates per-station (grouped) so rolling windows / shift don't leak
    across station boundaries."""
    out = []
    for station, g in daily.groupby("station", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        g["pm25_roll3"] = g["pm25"].rolling(3, min_periods=1).mean()
        g["pm25_roll7"] = g["pm25"].rolling(7, min_periods=1).mean()
        g["temp_roll3"] = g["temp"].rolling(3, min_periods=1).mean()
        g["month"] = g["date"].dt.month
        g["dow"] = g["date"].dt.dayofweek
        g["month_sin"], g["month_cos"] = _cyclic(g["month"], 12)
        g["dow_sin"], g["dow_cos"] = _cyclic(g["dow"], 7)

        # next-day target
        g["pm25_next"] = g["pm25"].shift(-1)
        g["aqi_cat_next"] = aqi_category(g["pm25_next"])
        g = g.iloc[:-1]  # last row has no next-day label
        out.append(g)
    result = pd.concat(out, ignore_index=True)
    result = result.dropna(subset=["aqi_cat_next"])
    return result


def build_dataset(city_df):
    feat = add_features_and_target(city_df)
    X = feat[COMMON_FEATURES].copy()
    y = feat["aqi_cat_next"].astype(int).values
    meta = feat[["city", "station", "date"]].copy()
    return X, y, meta


if __name__ == "__main__":
    bj = load_beijing_daily()
    dl = load_delhi_daily()
    print("Beijing daily:", bj.shape, bj.station.nunique(), "stations",
          bj.date.min().date(), "-", bj.date.max().date())
    print("Delhi daily:", dl.shape, dl.date.min().date(), "-", dl.date.max().date())

    Xb, yb, mb = build_dataset(bj)
    Xd, yd, md = build_dataset(dl)
    print("\nBeijing features:", Xb.shape, "target dist:",
          {AQI_LABELS[c]: int((yb == c).sum()) for c in sorted(set(yb))})
    print("Delhi features:", Xd.shape, "target dist:",
          {AQI_LABELS[c]: int((yd == c).sum()) for c in sorted(set(yd))})
    print("\nNaN fraction per feature (Beijing):")
    print(Xb.isna().mean()[Xb.isna().mean() > 0])
    print("\nNaN fraction per feature (Delhi):")
    print(Xd.isna().mean()[Xd.isna().mean() > 0])
    print("\nStations:", sorted(bj.station.unique()))
