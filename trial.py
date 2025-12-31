import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from dateutil import parser as dateutil_parser
import pytz
import time
import spacy
import re
import io

# ---------------- CONFIGURATION ----------------
st.set_page_config(page_title="Wellpack Weather", layout="wide")

# ---------------- SIDEBAR: PROJECT STATUS ----------------
st.sidebar.title("Project Status")
with st.sidebar.expander("✅ To-Do List Compliance", expanded=False):
    st.markdown("""
    **Core Features:**
    - [x] **Point Forecast** (Quick/Detailed)
    - [x] **Timezone Enforced** (Europe/Paris)

    **Batching Requirements:**
    - [x] **Multi-Zone Support** (Zip, Dept, Region)
    - [x] **Proposed Datetimes** (From CSV/Excel)
    - [x] **Full Metrics** (Tmin, Tmax, Rain, Snow)
    - [x] **Output Schema** (Standardized CSV)
    - [x] **Best 3 Hours Logic** (Auto-optimization)
    """)

# ---------------- Constants ----------------
TIMEZONE = "Europe/Paris"

# ---------------- Helper Functions (Original) ----------------
def geocode_french_zip(postal_code: str):
    """Return lat, lon, name for a French postal code via geo.api.gouv.fr"""
    try:
        url = f"https://geo.api.gouv.fr/communes?codePostal={postal_code}&fields=centre,nom&format=json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None, None, None
        coords = data[0]["centre"]["coordinates"]
        name = data[0]["nom"]
        lon, lat = coords
        return lat, lon, name
    except:
        return None, None, None


def fetch_hourly_for_datetimes(lat, lon, datetime_list):
    """
    Fetch hourly forecast only for the requested datetime points.
    """
    if not datetime_list:
        return pd.DataFrame()

    df_all = []
    for dt in datetime_list:
        start_str = dt.strftime("%Y-%m-%d")
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_str,
            "end_date": start_str,
            "hourly": "temperature_2m,precipitation,precipitation_probability,snow_depth,windspeed_10m,relativehumidity_2m,cloudcover",
            "timezone": TIMEZONE
        }
        try:
            r = requests.get(url, params=params, timeout=20)
            r.raise_for_status()
            data = r.json().get("hourly", {})
            if not data:
                continue
            df = pd.DataFrame(data)
            df["time"] = pd.to_datetime(df["time"])
            df_all.append(df)
        except:
            continue

    if not df_all:
        return pd.DataFrame()

    df_full = pd.concat(df_all).drop_duplicates(subset="time").sort_values("time").reset_index(drop=True)
    df_full["Date"] = df_full["time"].dt.date
    df_full["Time"] = df_full["time"].dt.time
    df_full.rename(columns={
        "temperature_2m": "Temperature (°C)",
        "precipitation": "Rain (mm)",
        "precipitation_probability": "Rain Prob (%)",
        "snow_depth": "Snow (cm)",
        "windspeed_10m": "Wind (km/h)",
        "relativehumidity_2m": "Humidity (%)",
        "cloudcover": "Cloud (%)"
    }, inplace=True)

    # Round numeric columns to 1 decimal
    for col in ["Temperature (°C)", "Rain (mm)", "Rain Prob (%)", "Snow (cm)",
                "Wind (km/h)", "Humidity (%)", "Cloud (%)"]:
        if col in df_full:
            df_full[col] = df_full[col].apply(lambda x: round(x, 1) if pd.notnull(x) else None)
    return df_full


def comfort_score(row):
    # Simple scoring: ideal temp ~20°C, low precipitation, moderate humidity, low wind/cloud
    score = 0
    temp = row.get("Temperature (°C)", 0.0)
    rain = row.get("Rain (mm)", 0.0)
    snow = row.get("Snow (cm)", 0.0)
    wind = row.get("Wind (km/h)", 0.0)
    cloud = row.get("Cloud (%)", 0.0)
    hum = row.get("Humidity (%)", 0.0)
    score += max(0, 20 - abs(temp - 20))
    score -= (rain + snow) * 2
    score -= max(0, abs(hum - 50) / 2)
    score -= wind * 0.5
    score -= cloud * 0.1
    return round(score, 1)


# ---------------- Helper Functions (Batching Pro) ----------------
def resolve_zone_batch(query, zone_type):
    """
    Advanced resolver for Batching (Supports Zip, Dept, Region, France, Extracted Location)
    Returns lat, lon, readable_name
    """
    query = str(query).strip()

    # A. Whole Country
    if zone_type == "France (Whole Country)" or query.upper() in ["FR", "FRANCE"]:
        return 46.603354, 1.888334, "France (National)"

    # B. Postal Code
    elif zone_type == "Postal Code":
        query = query.zfill(5)
        url = f"https://geo.api.gouv.fr/communes?codePostal={query}&fields=centre,nom&format=json"
        try:
            r = requests.get(url, timeout=5)
            data = r.json()
            if data:
                c = data[0]["centre"]["coordinates"]
                return c[1], c[0], f"{data[0]['nom']} ({query})"
        except:
            return None, None, None

    # C. Department
    elif zone_type == "Department":
        url = f"https://geo.api.gouv.fr/departements/{query}?fields=centre,nom"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                c = data["centre"]["coordinates"]
                return c[1], c[0], f"Dept: {data['nom']} ({query})"
        except:
            return None, None, None

    # D. Region
    elif zone_type == "Region":
        url = f"https://geo.api.gouv.fr/regions/{query}?fields=centre,nom"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                data = r.json()
                c = data["centre"]["coordinates"]
                return c[1], c[0], f"Region: {data['nom']} ({query})"
        except:
            return None, None, None

    # E. Extracted location (try multiple heuristics)
    elif zone_type == "Extracted Location":
        # 1) If a 5-digit postal code appears in the text, prioritize that:
        postal_search = re.search(r"\b(\d{5})\b", query)
        if postal_search:
            return resolve_zone_batch(postal_search.group(1), "Postal Code")

        # 2) Try searching communes by name (take last token after semicolon if present)
        try:
            candidate = query.split(";")[-1].strip()  # often "Store; City" -> take City
            if candidate:
                r = requests.get(f"https://geo.api.gouv.fr/communes?nom={requests.utils.requote_uri(candidate)}&fields=centre,nom&format=json", timeout=6)
                if r.status_code == 200:
                    data = r.json()
                    if data:
                        c = data[0]["centre"]["coordinates"]
                        return c[1], c[0], f"{data[0]['nom']} (resolved from '{query}')"
        except:
            pass

        # 3) Fall back to Nominatim (OpenStreetMap) search (last resort)
        try:
            nom_query = requests.utils.requote_uri(query)
            nom_url = f"https://nominatim.openstreetmap.org/search?q={nom_query}&format=json&limit=1&addressdetails=0"
            r = requests.get(nom_url, headers={"User-Agent": "WellpackWeather/1.0"}, timeout=8)
            if r.status_code == 200:
                data = r.json()
                if data:
                    lat = float(data[0]["lat"])
                    lon = float(data[0]["lon"])
                    display = data[0].get("display_name", query)
                    return lat, lon, f"{display} (OSM)"
        except:
            pass

        return None, None, None

    return None, None, None


def fetch_batch_weather(lat, lon, date):
    """
    Fetches full metrics (Daily Tmin/Tmax + Hourly) for Batch processing.
    """
    date_str = date.strftime("%Y-%m-%d")
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "temperature_2m,precipitation_probability,precipitation,snowfall,windspeed_10m",
        "daily": "temperature_2m_max,temperature_2m_min",
        "timezone": TIMEZONE
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        return r.json()
    except:
        return None


def calculate_batch_score(row):
    # Specialized scoring for batching logic
    score = 100
    score -= abs(row.get("temperature_2m", 20) - 20) * 2
    score -= row.get("precipitation", 0) * 5
    score -= row.get("windspeed_10m", 0) * 0.5
    return round(score, 1)


def format_output_row(zone_val, zone_type, loc_name, date_obj, daily, hourly_row, type_label):
    # Standardizes output schema
    snow_mm = (hourly_row.get("snowfall", 0) * 10) if pd.notnull(hourly_row.get("snowfall", 0)) else 0
    snow_pct = hourly_row.get("precipitation_probability", 0) if hourly_row.get("temperature_2m", 0) < 2 else 0

    return {
        "Zone Value": zone_val,
        "Zone Type": zone_type,
        "Location": loc_name,
        "Prediction Type": type_label,
        "Datetime": hourly_row.get("time"),
        "Temp (°C)": hourly_row.get("temperature_2m"),
        "Tmin (°C)": daily.get("temperature_2m_min", [None])[0],
        "Tmax (°C)": daily.get("temperature_2m_max", [None])[0],
        "Rain (mm)": hourly_row.get("precipitation"),
        "Rain (%)": hourly_row.get("precipitation_probability"),
        "Snow (mm)": snow_mm,
        "Snow (%)": snow_pct,
        "Wind (km/h)": hourly_row.get("windspeed_10m")
    }


# ---------------- NLP: spaCy French model ----------------
# Load the model once at startup
nlp = spacy.load("fr_core_news_lg")


def extract_location_from_message(text):
    """
    Extract locations from message text using spaCy French model.
    Returns a semi-normalized string (semicolon separated) if multiple found.
    """
    doc = nlp(str(text))
    locations = [ent.text for ent in doc.ents if ent.label_ in ("LOC", "GPE", "FAC")]
    if locations:
        # join them; downstream resolution will try heuristics
        return "; ".join(locations)
    return None


# ---------------- Streamlit UI ----------------
st.title("Wellpack Weather Prediction")
st.subheader("Forecast Tool")

# Initialize timezone object
tz = pytz.timezone(TIMEZONE)
current_time_paris = datetime.now(tz)

mode = st.radio("Forecast Mode", ["Quick Forecast", "Detailed Forecast", "Batch Processing (Pro CSV)"])
postal_code = ""

# Only show top-level postal input for Quick/Detailed modes (unchanged behavior)
if mode in ["Quick Forecast", "Detailed Forecast"]:
    postal_code = st.text_input("Enter French postal code (e.g. 94320)")

# ==========================================
# MODE 1: Quick Forecast (unchanged)
# ==========================================
if mode == "Quick Forecast":
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

    if st.button("Generate Forecast"):
        if not postal_code:
            st.error("Please enter postal code.")
            st.stop()

        lat, lon, place = geocode_french_zip(postal_code.strip())
        if lat is None:
            st.error("Invalid postal code.")
            st.stop()

        st.info(f"Fetching data for {place} ({lat}, {lon})…")

        df = fetch_hourly_for_datetimes(lat, lon, datetime_inputs)
        if df.empty:
            st.warning("No data available for the selected datetimes.")
        else:
            rows = []
            for dt in datetime_inputs:
                dt_naive = dt.replace(tzinfo=None)
                closest_row = df.iloc[(df['time'] - dt_naive).abs().argsort()[:1]]
                if closest_row.empty: continue
                row = closest_row.iloc[0]
                rows.append({
                    "Requested Date": dt.date(),
                    "Requested Time": dt.time(),
                    "Forecast Date": row["Date"],
                    "Forecast Hour": row["Time"],
                    "Temperature (°C)": row["Temperature (°C)"],
                    "Rain (mm)": row["Rain (mm)"],
                    "Rain Prob (%)": row["Rain Prob (%)"],
                    "Snow (cm)": row["Snow (cm)"],
                    "Wind (km/h)": row["Wind (km/h)"],
                    "Humidity (%)": row["Humidity (%)"],
                    "Cloud (%)": row["Cloud (%)"],
                })
            df_quick = pd.DataFrame(rows)
            st.dataframe(df_quick, use_container_width=True)

# ==========================================
# MODE 2: Detailed Forecast (unchanged)
# ==========================================
elif mode == "Detailed Forecast":
    st.markdown("Detailed Forecast — top 3 hours/day for selected date and next 4 days")
    start_date = st.date_input("Start date", value=current_time_paris.date())

    if st.button("Generate Forecast"):
        if not postal_code:
            st.error("Please enter postal code.")
            st.stop()

        lat, lon, place = geocode_french_zip(postal_code.strip())
        if lat is None:
            st.error("Invalid postal code.")
            st.stop()

        st.info(f"Fetching data for {place} ({lat}, {lon})…")

        # Original Detailed Logic
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = (start_date + timedelta(days=4)).strftime("%Y-%m-%d")
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon, "start_date": start_str, "end_date": end_str,
            "hourly": "temperature_2m,precipitation,precipitation_probability,snow_depth,windspeed_10m,relativehumidity_2m,cloudcover",
            "timezone": TIMEZONE
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            data = r.json().get("hourly", {})
            df_all = pd.DataFrame(data)
            df_all["time"] = pd.to_datetime(df_all["time"])
            df_all["Date"] = df_all["time"].dt.date
            df_all["Time"] = df_all["time"].dt.time
            df_all.rename(columns={
                "temperature_2m": "Temperature (°C)", "precipitation": "Rain (mm)",
                "precipitation_probability": "Rain Prob (%)", "snow_depth": "Snow (cm)",
                "windspeed_10m": "Wind (km/h)", "relativehumidity_2m": "Humidity (%)",
                "cloudcover": "Cloud (%)"
            }, inplace=True)
            for col in ["Temperature (°C)", "Rain (mm)", "Rain Prob (%)", "Snow (cm)", "Wind (km/h)", "Humidity (%)",
                        "Cloud (%)"]:
                if col in df_all: df_all[col] = df_all[col].apply(lambda x: round(x, 1) if pd.notnull(x) else None)
        except:
            st.error("Failed to fetch detailed forecast data.")
            st.stop()

        if df_all.empty:
            st.error("No data retrieved.")
            st.stop()

        df_all["Score"] = df_all.apply(comfort_score, axis=1)
        display_cols = ["Date", "Time", "Temperature (°C)", "Rain (mm)", "Rain Prob (%)", "Snow (cm)",
                        "Wind (km/h)", "Humidity (%)", "Cloud (%)", "Score"]

        for i in range(5):
            d = start_date + timedelta(days=i)
            day_df = df_all[df_all["Date"] == d]
            if day_df.empty: continue
            top3 = day_df.sort_values("Score", ascending=False).head(3)
            st.subheader(f"Top 3 Hours for {d}")
            st.dataframe(top3[display_cols])

        st.subheader("AI Suggested Best 3 Hours (From fetched data)")
        top3_ai = df_all.sort_values("Score", ascending=False).head(3)
        st.dataframe(top3_ai[display_cols])

# ==========================================
# MODE 3: Batch Processing (PRO) - Uploader + Column select + Run button
# ==========================================
elif mode == "Batch Processing (Pro CSV)":
    st.markdown("""
    **Universal Batch Loader.** Supports `postal_code`, `department`, `region` columns.
    - **With Time:** Forecasts specific hour.
    - **No Time:** Finds **Best 3 Hours** (optimized).
    - **Includes Tmax/Tmin** and full metrics.
    """)

    uploaded_file = st.file_uploader("Upload CSV or Excel (xlsx)", type=["csv", "xlsx"])
    # Only run selection UI AFTER upload. This ensures the dropdowns are clickable and not rendered inside a long-running loop.
    if uploaded_file:
        # Read file but do NOT start processing until user clicks the Run Batch button below
        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")
            st.stop()

        if df.empty:
            st.error("Uploaded file appears to be empty.")
            st.stop()

        # Use the dataframe's original column names for display; sort A-Z
        col_options = sorted(list(df.columns.astype(str)))
        col_display_hint = "Select the column that contains the field."

        st.info("Choose which columns contain the Message, Date, and Time. Columns shown A → Z.")
        message_col = st.selectbox("Message column (for NLP extraction)", ["-- None --"] + col_options, index=0)
        date_col = st.selectbox("Date column", ["-- None --"] + col_options, index=0)
        time_col = st.selectbox("Time column", ["-- None --"] + col_options, index=0)

        # PROCESS button: starts the long-running loop
        if st.button("Run Batch"):
            # Prepare normalized dataframe for internal usage
            df_proc = df.copy()
            # Ensure consistent access: keep original column names
            results = []
            bar = st.progress(0)

            # If message_col selected and not None placeholder, extract
            if message_col and message_col != "-- None --":
                st.info("Extracting locations from Message column using spaCy (fr_core_news_lg)...")
                df_proc["extracted_location"] = df_proc[message_col].apply(lambda x: extract_location_from_message(x) if pd.notnull(x) else None)
            else:
                df_proc["extracted_location"] = None

            # iterate rows
            for i, row in df_proc.iterrows():
                bar.progress((i + 1) / len(df_proc))

                # ---------------- SMART ZONE DETECTION ----------------
                z_val, z_type = None, "Postal Code"

                # Priority 1: Use extracted location from NLP if exists
                if row.get("extracted_location") and pd.notnull(row.get("extracted_location")):
                    z_val = row.get("extracted_location")
                    z_type = "Extracted Location"
                # Priority 2: existing columns (case-insensitive keys)
                elif any(c.lower() == "postal_code" or c.lower() == "postal code" for c in df_proc.columns):
                    # robust lookup
                    for c in df_proc.columns:
                        if c.lower().strip() in ("postal_code", "postal code", "zip", "postal"):
                            if pd.notnull(row.get(c)):
                                z_val = str(row.get(c))
                                z_type = "Postal Code"
                                break
                elif "department" in [c.lower().strip() for c in df_proc.columns]:
                    for c in df_proc.columns:
                        if c.lower().strip() == "department":
                            if pd.notnull(row.get(c)):
                                z_val = str(row.get(c)); z_type = "Department"; break
                elif "region" in [c.lower().strip() for c in df_proc.columns]:
                    for c in df_proc.columns:
                        if c.lower().strip() == "region":
                            if pd.notnull(row.get(c)):
                                z_val = str(row.get(c)); z_type = "Region"; break
                elif "zone" in [c.lower().strip() for c in df_proc.columns]:
                    for c in df_proc.columns:
                        if c.lower().strip() == "zone":
                            if pd.notnull(row.get(c)):
                                z_val = str(row.get(c)); z_type = row.get("type", "Postal Code"); break

                if not z_val:
                    results.append({"Zone Value": None, "Location": "Invalid/Not Found"})
                    continue

                # ---------------- RESOLVE ----------------
                lat, lon, loc_name = resolve_zone_batch(z_val, z_type)
                if not lat:
                    results.append({"Zone Value": z_val, "Location": "Invalid/Not Found"})
                    continue

                # ---------------- DATE/TIME ----------------
                d_str = row.get(date_col) if (date_col and date_col != "-- None --") else None
                t_str = row.get(time_col) if (time_col and time_col != "-- None --") else None
                try:
                    d_obj = dateutil_parser.parse(str(d_str)).date() if pd.notnull(d_str) else datetime.now().date()
                except:
                    d_obj = datetime.now().date()

                # ---------------- FETCH WEATHER ----------------
                data = fetch_batch_weather(lat, lon, d_obj)
                if not data:
                    # skip if no data returned
                    continue

                daily = data.get("daily", {})
                hourly_df = pd.DataFrame(data.get("hourly", {}))
                if not hourly_df.empty:
                    hourly_df["time"] = pd.to_datetime(hourly_df["time"])

                # ---------------- PROCESS: specific time or top3 ----------------
                if pd.notnull(t_str) and str(t_str).strip():
                    try:
                        t_obj = dateutil_parser.parse(str(t_str)).time()
                        tgt = datetime.combine(d_obj, t_obj)
                        hourly_df["diff"] = abs(hourly_df["time"] - tgt)
                        best_row = hourly_df.sort_values("diff").iloc[0]
                        results.append(format_output_row(z_val, z_type, loc_name, d_obj, daily, best_row, "Specific Time"))
                    except:
                        # if time parse fails, fallback to top3
                        hourly_df["score"] = hourly_df.apply(calculate_batch_score, axis=1)
                        top3 = hourly_df.sort_values("score", ascending=False).head(3)
                        rank = 1
                        for _, best_row in top3.iterrows():
                            out = format_output_row(z_val, z_type, loc_name, d_obj, daily, best_row, f"Best #{rank}")
                            out["Comfort Score"] = best_row["score"]
                            results.append(out)
                            rank += 1
                else:
                    hourly_df["score"] = hourly_df.apply(calculate_batch_score, axis=1)
                    top3 = hourly_df.sort_values("score", ascending=False).head(3)
                    rank = 1
                    for _, best_row in top3.iterrows():
                        out = format_output_row(z_val, z_type, loc_name, d_obj, daily, best_row, f"Best #{rank}")
                        out["Comfort Score"] = best_row["score"]
                        results.append(out)
                        rank += 1

                # small sleep to be polite with external APIs
                time.sleep(0.02)

            # ---------------- OUTPUT ----------------
            if results:
                st.success("Batch Complete!")
                res_df = pd.DataFrame(results)
                display_order = ["Zone Value", "Zone Type", "Location", "Datetime", "Prediction Type",
                                 "Tmin (°C)", "Tmax (°C)", "Temp (°C)", "Rain (mm)", "Rain (%)", "Snow (mm)", "Snow (%)", "Comfort Score"]
                final_cols = [c for c in display_order if c in res_df.columns]
                st.dataframe(res_df[final_cols], use_container_width=True)

                # Download as CSV
                csv_bytes = res_df[final_cols].to_csv(index=False).encode("utf-8")
                st.download_button("Download results as CSV", csv_bytes, file_name="weather_batch_results.csv", mime="text/csv")

                # Download as Excel
                towrite = io.BytesIO()
                with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                    res_df[final_cols].to_excel(writer, index=False, sheet_name="results")
                towrite.seek(0)
                st.download_button("Download results as Excel", towrite.read(), file_name="weather_batch_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            else:
                st.warning("No valid data processed or no results returned.")