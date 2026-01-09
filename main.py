import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date, timedelta
from dateutil import parser as dateutil_parser
import pytz
import time
import spacy
import re
import io
import unicodedata

# =========================================================
# ---------------- CONFIGURATION ----------------
# =========================================================
st.set_page_config(page_title="Wellpack Weather + Shoot Extractor", layout="wide")

TIMEZONE = "Europe/Paris"
tz = pytz.timezone(TIMEZONE)
current_time_paris = datetime.now(tz)

# =========================================================
# ---------------- Load spaCy once ----------------
# =========================================================
@st.cache_resource
def load_spacy():
    return spacy.load("fr_core_news_lg")

nlp = load_spacy()

# =========================================================
# ---------------- French datetime mapping ----------------
# =========================================================
FRENCH_MONTHS = {
    "janvier": "january", "février": "february", "fevrier": "february",
    "mars": "march", "avril": "april", "mai": "may", "juin": "june",
    "juillet": "july", "août": "august", "aout": "august",
    "septembre": "september", "octobre": "october", "novembre": "november",
    "décembre": "december", "decembre": "december"
}
FRENCH_WEEKDAYS = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

# =========================================================
# ---------------- SAFE CSV LOADER ----------------
# =========================================================
def read_csv_safe(uploaded_file):
    encodings = ["utf-8", "cp1252", "latin1"]
    seps = [",", ";", "\t", "|"]

    for enc in encodings:
        for sep in seps:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=enc, sep=sep)
                if df.shape[1] > 1:
                    return df
            except:
                continue

    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, encoding="latin1", sep=";")

# =========================================================
# ---------------- NORMALIZATION ----------------
# =========================================================
def normalize_text(s):
    if not s:
        return ""
    s = str(s).lower().replace("-", " ")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9\s']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# =========================================================
# ---------------- STORE CODE ----------------
# =========================================================
STORE_CODE_RE = re.compile(r"\[\[([A-Za-z0-9]{1,6})\]\]")

def extract_store_code(msg):
    if not msg:
        return None
    m = STORE_CODE_RE.search(str(msg))
    if m:
        return m.group(1).strip().upper()
    return None

# =========================================================
# ---------------- ADDRESS DETECTION ----------------
# =========================================================
def message_has_address(msg):
    if not msg:
        return False
    m = normalize_text(msg)
    return any(w in m for w in [
        "rue", "bd", "boulevard", "avenue", "zac", "cc", "c c",
        "centre commercial", "place", "pl "
    ])

# =========================================================
# ---------------- CITY EXTRACTION RULES ----------------
# =========================================================
PREP_CITY_PATTERN = r"\b(?:à|a|au|aux)\s+([A-ZÀ-Ÿ][A-ZÀ-Ÿ'\- ]{3,})"
PLUS_CITY_PATTERN = r"\+\s*([A-ZÀ-Ÿ][A-ZÀ-Ÿ'\- ]{3,})"

# =========================================================
# ---------------- CLEAN CITY ----------------
# =========================================================
def clean_candidate_city(text):
    if not text:
        return None

    text = str(text).strip()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = text.replace("[[RICH]]", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.split(r"[,.!?:;]", text)[0]

    STOP_WORDS = [
        "déménage", "demenage", "déménagé", "demenagé",
        "réouvre", "reouvre", "réouverture", "reouverture",
        "ouvre", "ouverture", "rénové", "renove", "travaux",
        "retrouvez", "retrouve", "decouvrez", "découvrez",
        "beneficiez", "bénéficiez", "profitez", "offre", "offres",
        "fidélité", "fidelite", "jusqu", "concept",
        "a", "à", "au", "aux", "de", "du", "des", "d'",
        "rue", "bd", "boulevard", "avenue", "zac", "zone", "centre", "pl", "place"
    ]

    txt_norm = normalize_text(text)
    for w in STOP_WORDS:
        w_norm = normalize_text(w)
        if f" {w_norm} " in f" {txt_norm} ":
            out_tokens = []
            for tok in text.split():
                if normalize_text(tok) == w_norm:
                    break
                out_tokens.append(tok)
            text = " ".join(out_tokens)
            break

    text = re.sub(r"[^A-Za-zÀ-ÿ\-\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) < 3:
        return None
    return text.title()

def extract_uppercase_city_single(msg: str):
    if not msg:
        return None
    words = re.findall(r"\b[A-ZÀ-Ÿ]{4,}(?:-[A-ZÀ-Ÿ]{2,})*\b", msg)
    bad = {"RDV", "PROMO", "OFFRES", "SMS", "RICH", "ICI", "NOUVEAU", "ERRATUM"}
    words = [w for w in words if w not in bad]
    if words:
        best = max(words, key=len)
        return clean_candidate_city(best)
    return None

# =========================================================
# ---------------- LOCATION EXTRACTION ----------------
# =========================================================
BRAND_PATTERNS = [
    r"(?:écouter\s*voir|ecouter\s*voir)\s*(?:optique\s*mutualiste\s*)?de\s+([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\s']{2,})",
    r"(?:écouter\s*voir|ecouter\s*voir)\s+([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\s']{2,})",
    r"marie\s*blach[eè]re\s+([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\s']{2,})",
    r"dacia\s+([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\s']{2,})",
    r"st\s*maclou\s+([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\s']{2,})",
    r"districenter\s+([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\s']{2,})",
    r"carrefour\s*(?:express|city|market|contact)?\s+([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\s']{2,})",
    r"gedimat\s*\+?\s*([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\s']{2,})",
    r"magic\s*form\s+([A-ZÀ-ÿ][A-Za-zÀ-ÿ\-\s']{2,})",
]

def extract_location_smart(message: str):
    if pd.isna(message):
        return None, None

    msg = str(message)
    postal_match = re.search(r"\b\d{5}\b", msg)
    postal = postal_match.group() if postal_match else None

    mprep = re.search(PREP_CITY_PATTERN, msg)
    if mprep:
        cand = clean_candidate_city(mprep.group(1))
        if cand:
            return cand, postal

    mplus = re.search(PLUS_CITY_PATTERN, msg)
    if mplus:
        cand = clean_candidate_city(mplus.group(1))
        if cand:
            return cand, postal

    for pat in BRAND_PATTERNS:
        m = re.search(pat, msg, flags=re.IGNORECASE)
        if m:
            cand = clean_candidate_city(m.group(1))
            if cand:
                return cand, postal

    doc = nlp(msg)
    locs = [ent.text for ent in doc.ents if ent.label_ in ("LOC", "GPE", "FAC")]
    locs = [clean_candidate_city(l) for l in locs if clean_candidate_city(l)]
    if locs:
        cand = max(locs, key=len)
        return cand, postal

    up = extract_uppercase_city_single(msg)
    if up:
        return up, postal

    return None, postal

# =========================================================
# ---------------- GEO HELPERS ----------------
# =========================================================
@st.cache_data(show_spinner=False)
def geocode_french_zip(postal_code: str):
    try:
        url = f"https://geo.api.gouv.fr/communes?codePostal={postal_code}&fields=centre,nom,codesPostaux&format=json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None, None, None, None
        coords = data[0]["centre"]["coordinates"]
        name = data[0]["nom"]
        postals = data[0].get("codesPostaux", [postal_code])
        lon, lat = coords
        return lat, lon, name, postals[0] if postals else postal_code
    except:
        return None, None, None, None

@st.cache_data(show_spinner=False)
def validate_french_city(city_name: str):
    try:
        if not city_name:
            return False, None, None, None, None
        city_name = clean_candidate_city(city_name)
        if not city_name:
            return False, None, None, None, None

        url_nom = f"https://geo.api.gouv.fr/communes?nom={requests.utils.requote_uri(city_name)}&fields=centre,nom,codesPostaux&boost=population&format=json"
        r = requests.get(url_nom, timeout=10)
        r.raise_for_status()
        data = r.json()
        if data:
            best = data[0]
            lon, lat = best["centre"]["coordinates"]
            postal = best.get("codesPostaux", [None])[0]
            return True, lat, lon, best["nom"], postal
        return False, None, None, None, None
    except:
        return False, None, None, None, None

@st.cache_data(show_spinner=False)
def geocode_from_message_ban(msg: str, fallback_city=None, fallback_postal=None):
    try:
        if not msg:
            return None, None, None, None, None, 0
        q = re.sub(r"https?://\S+", " ", str(msg))
        q = q.replace("[[RICH]]", " ")
        q = re.sub(r"\s+", " ", q).strip()

        hints = []
        if fallback_city:
            hints.append(str(fallback_city))
        if fallback_postal:
            hints.append(str(fallback_postal))
        full_query = " ".join(hints + [q]).strip()

        url = "https://api-adresse.data.gouv.fr/search/"
        params = {"q": full_query, "limit": 1}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data.get("features"):
            return None, None, None, None, None, 0

        feat = data["features"][0]
        props = feat["properties"]
        coords = feat["geometry"]["coordinates"]

        lon, lat = coords[0], coords[1]
        label = props.get("label")
        postcode = props.get("postcode")
        city = props.get("city")
        score = props.get("score", 0)

        return lat, lon, label, postcode, city, score
    except:
        return None, None, None, None, None, 0

# =========================================================
# ---------------- WEATHER FETCH (Hourly + Daily) ----------------
# =========================================================
@st.cache_data(show_spinner=False)
def fetch_weather_with_daily(lat, lon, start_date, end_date, is_past: bool):
    url = "https://archive-api.open-meteo.com/v1/archive" if is_past else "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,precipitation_probability,precipitation,snowfall,windspeed_10m,relativehumidity_2m,cloudcover",
        "daily": "temperature_2m_min,temperature_2m_max",
        "timezone": TIMEZONE
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

def get_nearest_hour(hourly_df, tgt_dt):
    diffs = (hourly_df["time"] - tgt_dt).abs()
    best_idx = diffs.idxmin()
    return hourly_df.loc[best_idx]

def extract_metrics(data, target_dt):
    hourly_df = pd.DataFrame(data.get("hourly", {}))
    hourly_df["time"] = pd.to_datetime(hourly_df["time"])

    daily = data.get("daily", {})

    row = get_nearest_hour(hourly_df, target_dt)

    tmin = daily.get("temperature_2m_min", [None])[0]
    tmax = daily.get("temperature_2m_max", [None])[0]

    rain_pct = row.get("precipitation_probability")
    rain_mm = row.get("precipitation")
    snow_mm = row.get("snowfall")
    temp = row.get("temperature_2m")
    wind = row.get("windspeed_10m")

    snow_pct = rain_pct if (temp is not None and temp <= 2) else 0

    return {
        "Forecast Time": row.get("time"),
        "Temp (°C)": temp,
        "Wind (km/h)": wind,
        "Tmin (°C)": tmin,
        "Tmax (°C)": tmax,
        "Rain (%)": rain_pct,
        "Rain (mm)": rain_mm,
        "Snow (%)": snow_pct,
        "Snow (mm)": snow_mm
    }

def compute_best_hour(hourly_df):
    df = hourly_df.copy()
    df["temperature_2m"] = pd.to_numeric(df["temperature_2m"], errors="coerce")
    df["precipitation"] = pd.to_numeric(df["precipitation"], errors="coerce").fillna(0)
    df["precipitation_probability"] = pd.to_numeric(df["precipitation_probability"], errors="coerce").fillna(0)
    df["windspeed_10m"] = pd.to_numeric(df["windspeed_10m"], errors="coerce").fillna(0)

    df["score"] = (
        100
        - abs(df["temperature_2m"] - 20) * 2
        - df["precipitation"] * 5
        - df["windspeed_10m"] * 0.5
        - df["precipitation_probability"] * 0.3
    )
    return df.sort_values("score", ascending=False).iloc[0]

def comfort_score(row):
    score = 0
    temp = row.get("temperature_2m", 0.0)
    rain = row.get("precipitation", 0.0)
    wind = row.get("windspeed_10m", 0.0)
    cloud = row.get("cloudcover", 0.0)
    hum = row.get("relativehumidity_2m", 0.0)

    score += max(0, 20 - abs(temp - 20))
    score -= rain * 2
    score -= max(0, abs(hum - 50) / 2)
    score -= wind * 0.5
    score -= cloud * 0.1
    return round(score, 1)

# =========================================================
# ---------------- DATETIME PARSE ----------------
# =========================================================
def parse_datetime_safe(dt_val):
    try:
        if pd.isnull(dt_val) or str(dt_val).strip() == "":
            return None, None
        if isinstance(dt_val, (datetime, pd.Timestamp)):
            return dt_val.date(), dt_val.time()

        s = str(dt_val).strip().lower()
        for wd in FRENCH_WEEKDAYS:
            s = s.replace(wd, "")
        s = s.replace("à", " ")
        for fr, en in FRENCH_MONTHS.items():
            s = s.replace(fr, en)

        s = re.sub(r"\s+", " ", s).strip()
        dt = dateutil_parser.parse(s, fuzzy=True)
        return dt.date(), dt.time()
    except:
        return None, None

# =========================================================
# --------------------------- UI --------------------------
# =========================================================
st.title("🌦 Wellpack Weather + Shoot Weather Extractor")

mode = st.radio("Select Mode", ["Quick Forecast", "Detailed Forecast", "Shoot Weather Extractor (CSV/Excel)"])

# =========================================================
# MODE 1: QUICK FORECAST
# =========================================================
if mode == "Quick Forecast":
    st.subheader("⚡ Quick Forecast (Postal Code + 5 datetime points)")
    postal_code = st.text_input("Enter French postal code (e.g. 94320)")

    st.markdown("Select up to 5 dates and times:")
    datetime_inputs = []
    cols_date = st.columns(5)
    cols_time = st.columns(5)

    for i in range(5):
        with cols_date[i]:
            d = st.date_input(f"Date {i + 1}", value=current_time_paris.date(), key=f"q_date_{i}")
        with cols_time[i]:
            t = st.time_input(f"Time {i + 1}", value=current_time_paris.time().replace(second=0, microsecond=0),
                              key=f"q_time_{i}")
        datetime_inputs.append(datetime.combine(d, t))

    if st.button("Generate Quick Forecast"):
        if not postal_code:
            st.error("Please enter postal code.")
            st.stop()

        lat, lon, place, _ = geocode_french_zip(postal_code.strip())
        if lat is None:
            st.error("Invalid postal code.")
            st.stop()

        rows = []
        for dt in datetime_inputs:
            date_str = dt.strftime("%Y-%m-%d")
            data = fetch_weather_with_daily(lat, lon, date_str, date_str, is_past=False)
            if not data:
                continue
            metrics = extract_metrics(data, dt)

            rows.append({"Requested Datetime": dt, "Location": place, **metrics})

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.warning("No forecast data found for chosen datetimes.")

# =========================================================
# MODE 2: DETAILED FORECAST
# =========================================================
elif mode == "Detailed Forecast":
    st.subheader("📊 Detailed Forecast (Top 3 hours/day for 5 days)")
    postal_code = st.text_input("Enter French postal code (e.g. 94320)")
    start_date = st.date_input("Start date", value=current_time_paris.date())

    if st.button("Generate Detailed Forecast"):
        if not postal_code:
            st.error("Please enter postal code.")
            st.stop()

        lat, lon, place, _ = geocode_french_zip(postal_code.strip())
        if lat is None:
            st.error("Invalid postal code.")
            st.stop()

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = (start_date + timedelta(days=4)).strftime("%Y-%m-%d")

        data = fetch_weather_with_daily(lat, lon, start_str, end_str, is_past=False)
        if not data or "hourly" not in data:
            st.error("Failed to fetch forecast.")
            st.stop()

        hourly_df = pd.DataFrame(data["hourly"])
        hourly_df["time"] = pd.to_datetime(hourly_df["time"])
        hourly_df["Date"] = hourly_df["time"].dt.date
        hourly_df["Time"] = hourly_df["time"].dt.time

        daily = data.get("daily", {})
        dates_daily = pd.to_datetime(daily["time"]).date
        tmins = daily["temperature_2m_min"]
        tmaxs = daily["temperature_2m_max"]

        tmin_map = dict(zip(dates_daily, tmins))
        tmax_map = dict(zip(dates_daily, tmaxs))

        hourly_df["Tmin (°C)"] = hourly_df["Date"].map(tmin_map)
        hourly_df["Tmax (°C)"] = hourly_df["Date"].map(tmax_map)

        hourly_df["Rain (%)"] = hourly_df["precipitation_probability"]
        hourly_df["Rain (mm)"] = hourly_df["precipitation"]
        hourly_df["Snow (mm)"] = hourly_df["snowfall"]
        hourly_df["Snow (%)"] = hourly_df.apply(lambda r: r["Rain (%)"] if r["temperature_2m"] <= 2 else 0, axis=1)

        hourly_df["Score"] = hourly_df.apply(comfort_score, axis=1)

        display_cols = ["Date", "Time", "Tmin (°C)", "Tmax (°C)", "temperature_2m",
                        "Rain (mm)", "Rain (%)", "Snow (mm)", "Snow (%)",
                        "windspeed_10m", "Score"]

        hourly_df.rename(columns={
            "temperature_2m": "Temp (°C)",
            "windspeed_10m": "Wind (km/h)"
        }, inplace=True)

        display_cols = ["Date", "Time", "Tmin (°C)", "Tmax (°C)", "Temp (°C)",
                        "Rain (mm)", "Rain (%)", "Snow (mm)", "Snow (%)",
                        "Wind (km/h)", "Score"]

        for i in range(5):
            d = start_date + timedelta(days=i)
            day_df = hourly_df[hourly_df["Date"] == d]
            if day_df.empty:
                continue
            top3 = day_df.sort_values("Score", ascending=False).head(3)
            st.subheader(f"Top 3 Hours for {d} ({place})")
            st.dataframe(top3[display_cols], use_container_width=True)

        st.subheader("Overall Best 3 Hours (All 5 days)")
        st.dataframe(hourly_df.sort_values("Score", ascending=False).head(3)[display_cols], use_container_width=True)

# =========================================================
# MODE 3: SHOOT WEATHER EXTRACTOR
# =========================================================
else:
    st.subheader("📌 Shoot Weather Extractor (CSV/Excel)")
    st.write("Upload messages + shoot datetime → extract location → resolve → fetch weather → export Excel.")

    uploaded_file = st.file_uploader("1) Upload Message CSV or Excel", type=["csv", "xlsx"])
    mapping_file = st.file_uploader("2) Upload Store Code Mapping CSV/Excel (optional)", type=["csv", "xlsx"])

    store_map = {}
    if mapping_file:
        if mapping_file.name.lower().endswith(".csv"):
            map_df = read_csv_safe(mapping_file)
        else:
            map_df = pd.read_excel(mapping_file)

        map_df.columns = [c.strip().lower() for c in map_df.columns]
        if "store_code" not in map_df.columns:
            st.error("❌ Mapping file must contain column: store_code")
        else:
            for _, row in map_df.iterrows():
                code = str(row.get("store_code")).strip().upper()
                store_map[code] = {
                    "address": row.get("address"),
                    "postal": str(row.get("postal")).strip() if pd.notnull(row.get("postal")) else None,
                    "city": row.get("city")
                }
            st.success(f"✅ Loaded {len(store_map)} store codes")

    if uploaded_file:
        if uploaded_file.name.lower().endswith(".csv"):
            df = read_csv_safe(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.success(f"✅ Loaded {len(df):,} rows")
        cols = sorted(list(df.columns.astype(str)))
        message_col = st.selectbox("Message column", cols)
        datetime_col = st.selectbox("Shoot datetime column", cols)

        remove_no_location = st.checkbox("🧹 Remove rows with NO extracted location", value=False)
        remove_invalid_final = st.checkbox("🚫 Remove rows Invalid after resolution", value=False)

        if st.button("🚀 Run Extraction & Weather"):
            df_proc = df.copy()

            df_proc[["shoot_date", "shoot_time"]] = df_proc[datetime_col].apply(lambda x: pd.Series(parse_datetime_safe(x)))
            df_proc["datetime_valid"] = df_proc["shoot_date"].notnull()

            df_proc["store_code"] = df_proc[message_col].apply(extract_store_code)
            df_proc[["extracted_city", "extracted_postal"]] = df_proc[message_col].apply(
                lambda x: pd.Series(extract_location_smart(x))
            )

            results = []
            today = date.today()
            bar = st.progress(0)
            total = len(df_proc)

            for i, row in enumerate(df_proc.itertuples(index=True)):
                bar.progress((i + 1) / total)
                r = df_proc.loc[row.Index]

                msg = r.get(message_col)
                store_code = r.get("store_code")
                extracted_city = clean_candidate_city(r.get("extracted_city"))
                extracted_postal = r.get("extracted_postal")

                if remove_no_location and not extracted_city and not extracted_postal and not store_code and not message_has_address(msg):
                    continue

                if not bool(r.get("datetime_valid")):
                    if remove_invalid_final:
                        continue
                    results.append({"Message": msg, "Resolved Location": "Invalid datetime"})
                    continue

                d_obj, t_obj = r.get("shoot_date"), r.get("shoot_time")
                tgt = datetime.combine(d_obj, t_obj)
                date_str = d_obj.strftime("%Y-%m-%d")
                is_past = d_obj < today

                lat, lon = None, None
                resolved_location, resolved_postal = None, None
                resolved_method, ban_score = None, None

                # City
                if extracted_city:
                    ok, lat2, lon2, official, post = validate_french_city(extracted_city)
                    if ok and lat2 and lon2:
                        lat, lon = lat2, lon2
                        resolved_location = official
                        resolved_postal = post
                        resolved_method = "city"

                # Postal
                if (not lat or not lon) and extracted_postal:
                    lat, lon, resolved_location, resolved_postal = geocode_french_zip(str(extracted_postal).zfill(5))
                    if lat and lon:
                        resolved_method = "postal"

                # Store mapping
                if (not lat or not lon) and store_code and store_code in store_map:
                    m = store_map[store_code]
                    map_city = clean_candidate_city(m.get("city"))
                    map_postal = m.get("postal")
                    map_addr = m.get("address")

                    if map_city:
                        ok, lat2, lon2, official, post = validate_french_city(map_city)
                        if ok and lat2 and lon2:
                            lat, lon = lat2, lon2
                            resolved_location = official
                            resolved_postal = post
                            resolved_method = "store_mapping_city"

                    if (not lat or not lon) and map_postal:
                        lat, lon, resolved_location, resolved_postal = geocode_french_zip(str(map_postal).zfill(5))
                        if lat and lon:
                            resolved_method = "store_mapping_postal"

                    if (not lat or not lon) and map_addr:
                        ban_lat, ban_lon, ban_label, ban_postal, ban_city, score = geocode_from_message_ban(
                            f"{map_addr} {map_city or ''} {map_postal or ''}",
                            fallback_city=map_city,
                            fallback_postal=map_postal
                        )
                        if ban_lat and ban_lon and score >= 0.50:
                            lat, lon = ban_lat, ban_lon
                            resolved_location = ban_label or ban_city or map_city
                            resolved_postal = ban_postal or map_postal
                            resolved_method = "store_mapping_BAN"
                            ban_score = score

                # BAN last
                if not lat or not lon:
                    ban_lat, ban_lon, ban_label, ban_postal, ban_city, score = geocode_from_message_ban(
                        msg,
                        fallback_city=extracted_city,
                        fallback_postal=extracted_postal
                    )
                    if ban_lat and ban_lon and score >= 0.50:
                        lat, lon = ban_lat, ban_lon
                        resolved_location = ban_label or ban_city or extracted_city
                        resolved_postal = ban_postal
                        resolved_method = "BAN"
                        ban_score = score

                if not lat or not lon:
                    if remove_invalid_final:
                        continue
                    results.append({
                        "Message": msg,
                        "Shoot Datetime": tgt,
                        "Store Code": store_code,
                        "Extracted City": extracted_city,
                        "Extracted Postal": extracted_postal,
                        "Resolved Location": "Invalid (Cannot resolve)",
                        "Resolved Method": resolved_method
                    })
                    continue

                data = fetch_weather_with_daily(lat, lon, date_str, date_str, is_past)
                if not data:
                    if remove_invalid_final:
                        continue
                    results.append({
                        "Message": msg,
                        "Shoot Datetime": tgt,
                        "Resolved Location": resolved_location,
                        "Resolved Postal": resolved_postal,
                        "Resolved Method": resolved_method
                    })
                    continue

                metrics = extract_metrics(data, tgt)

                # best hour
                hourly_df = pd.DataFrame(data.get("hourly", {}))
                hourly_df["time"] = pd.to_datetime(hourly_df["time"])
                best_row = compute_best_hour(hourly_df)

                results.append({
                    "Message": msg,
                    "Shoot Datetime": tgt,
                    "Extracted City": extracted_city,
                    "Extracted Postal": extracted_postal,
                    "Resolved Location": resolved_location,
                    "Resolved Postal": resolved_postal,
                    "Resolved Method": resolved_method,
                    **metrics,
                    "Best Time of Day": best_row.get("time"),
                    "Best Temp (°C)": best_row.get("temperature_2m")
                })

            st.success("✅ Completed!")
            res_df = pd.DataFrame(results)
            st.dataframe(res_df.head(200), use_container_width=True)

            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                res_df.to_excel(writer, index=False, sheet_name="shoot_weather")
            towrite.seek(0)

            st.download_button(
                "📥 Download Excel Output",
                towrite.read(),
                file_name="shoot_weather_output_FINAL.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

