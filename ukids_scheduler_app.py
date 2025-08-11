# Streamlit app for uKids Children Scheduler
# Save as ukids_scheduler_app.py and run with: streamlit run ukids_scheduler_app.py

import io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date
from typing import Dict, List, Tuple
from collections import defaultdict
from datetime import datetime
from io import BytesIO
import base64
import re

st.set_page_config(page_title='uKids Scheduler', layout='wide')
st.title('uKids Scheduler')

# Set black theme using custom CSS
st.markdown(
    """
    <style>
        body {
            background-color: #000000;
            color: white;
        }
        .stApp {
            background-color: #000000;
        }
        .stButton>button, .stDownloadButton>button {
            background-color: #444;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and center the logo
try:
    with open("image(1).png", "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
            <div style='text-align: center;'>
                <img src='data:image/png;base64,{encoded}' width='600'>
            </div>
        """, unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("Logo image not found. Please upload or place 'image.png' in the app directory.")

# ------------------------------
# Helpers
# ------------------------------
MONTHS = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12
}

def detect_name_column(df: pd.DataFrame) -> str:
    prefs = ['Name', 'Full name', 'Full names', 'What is your name AND surname?']
    for pref in prefs:
        for c in df.columns:
            if isinstance(c, str) and c.strip().lower() == pref.strip().lower():
                return c
    candidates = [c for c in df.columns if isinstance(c, str) and 'name' in c.lower()]
    if candidates:
        return candidates[0]
    return df.columns[0]

def is_priority_col(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors='coerce').dropna()
    if len(vals) == 0:
        return False
    return (vals.min() >= 0) and (vals.max() <= 5)

def build_long_df(people_df: pd.DataFrame, name_col: str, role_cols: List[str]) -> pd.DataFrame:
    records = []
    for _, r in people_df.iterrows():
        person = str(r[name_col]).strip()
        if not person or person.lower() == 'nan':
            continue
        for role in role_cols:
            pr = pd.to_numeric(r[role], errors='coerce')
            if pd.isna(pr):
                continue
            pr = int(round(pr))
            if pr >= 1:  # only roles with priority 1..5
                records.append({'person': person, 'role': role, 'priority': pr})
    return pd.DataFrame(records)

def parse_month_and_dates_from_headers(responses_df: pd.DataFrame) -> Tuple[int, int, Dict[str, pd.Timestamp]]:
    """
    Returns (year, month, date_map) where date_map maps availability column -> pd.Timestamp.
    We expect headers like 'Are you available 7 September?'.
    Year is inferred from the 'Timestamp' column if present; otherwise, current year.
    Validates that all availability columns belong to the same month.
    """
    avail_cols = [c for c in responses_df.columns if isinstance(c, str) and c.strip().lower().startswith('are you available')]
    if not avail_cols:
        raise ValueError("No 'Are you available ...' columns found in the responses.")
    # Extract month name and day from each column
    months_found = []
    days = []
    col_info = []
    for c in avail_cols:
        c_low = c.lower()
        # find month name
        month_name = None
        for m in MONTHS.keys():
            if m in c_low:
                month_name = m
                break
        # find first integer as day
        day_match = re.search(r'(\d{1,2})', c_low)
        day = int(day_match.group(1)) if day_match else None
        if not month_name or day is None:
            continue
        months_found.append(month_name)
        days.append(day)
        col_info.append((c, MONTHS[month_name], day))
    if not col_info:
        raise ValueError("Could not parse day/month from availability headers. Expected e.g. 'Are you available 7 September?'")
    # Ensure single month
    month_nums = {m for _, m, _ in col_info}
    if len(month_nums) > 1:
        raise ValueError(f"Multiple months detected in availability headers: {month_nums}. Upload one month at a time.")
    month = month_nums.pop()
    # Infer year
    if 'Timestamp' in responses_df.columns:
        # Use the most common year in Timestamp
        ts_years = pd.to_datetime(responses_df['Timestamp'], errors='coerce').dt.year.dropna().astype(int)
        year = int(ts_years.mode().iloc[0]) if not ts_years.empty else date.today().year
    else:
        year = date.today().year
    # Build map
    date_map = {c: pd.Timestamp(datetime(year, month, d)).normalize() for c, _, d in col_info}
    return year, month, date_map

def parse_availability(responses_df: pd.DataFrame, name_col_resp: str, year: int, month: int, date_map: Dict[str, pd.Timestamp]):
    availability: Dict[str, Dict[pd.Timestamp, bool]] = {}
    for _, row in responses_df.iterrows():
        nm = str(row.get(name_col_resp, '')).strip()
        if not nm or nm.lower() == 'nan':
            continue
        availability.setdefault(nm, {})
        for col, dt in date_map.items():
            ans = str(row.get(col, '')).strip().lower()
            is_yes = ans in {'yes', 'y', 'true', 'available'}
            availability[nm][dt] = is_yes
    service_dates = sorted(set(date_map.values()))
    return availability, service_dates

def schedule_exact2(long_df: pd.DataFrame, availability: Dict[str, Dict[pd.Timestamp, bool]], service_dates: List[pd.Timestamp], roles: List[str]):
    """
    Greedy two-pass scheduler for EXACTLY 2 per person where feasible:
    - Only assigns to roles with priority>=1
    - Respects availability
    - Avoids double-booking a person on the same date
    - Tries to give everyone 2 slots; if not possible, leaves slots unfilled or people under 2
    """
    rng = np.random.default_rng(123)
    assign_count = {p: 0 for p in long_df['person'].unique()}
    schedule = pd.DataFrame(index=service_dates, columns=roles)

    # Build per-slot eligibility
    slots = [(d, role) for d in service_dates for role in roles]
    slot_eligible = {}
    for (d, role) in slots:
        sub = long_df[long_df['role'] == role]
        elig = []
        for _, rr in sub.iterrows():
            p = rr['person']
            pr = rr['priority']
            if availability.get(p, {}).get(d, False):
                elig.append((p, pr))
        slot_eligible[(d, role)] = elig

    target = 2

    # Pass 1: prioritize people furthest from target (0 first), then higher priority
    for d in service_dates:
        assigned_today = set()
        for role in roles:
            elig = [(p, pr) for (p, pr) in slot_eligible[(d, role)]
                    if assign_count[p] < target and p not in assigned_today]
            if not elig:
                continue
            min_count = min(assign_count[p] for p, _ in elig)
            pool = [(p, pr) for (p, pr) in elig if assign_count[p] == min_count]
            pool.sort(key=lambda x: (-x[1], rng.random()))
            chosen = pool[0][0]
            schedule.loc[d, role] = chosen
            assign_count[chosen] += 1
            assigned_today.add(chosen)

    # Pass 2: fill remaining empties with under-target only
    for d in service_dates:
        assigned_today = set([v for v in schedule.loc[d, :].dropna().values])
        for role in roles:
            if pd.notna(schedule.loc[d, role]):
                continue
            elig = [(p, pr) for (p, pr) in slot_eligible[(d, role)]
                    if assign_count[p] < target and p not in assigned_today]
            if not elig:
                continue
            elig.sort(key=lambda x: (assign_count[x[0]], -x[1], rng.random()))
            for p, pr in elig:
                if assign_count[p] < target:
                    schedule.loc[d, role] = p
                    assign_count[p] += 1
                    assigned_today.add(p)
                    break

    return schedule, assign_count

def make_report(assign_count, availability, service_dates):
    series = pd.Series(assign_count, name='AssignedCount').sort_values(ascending=False)
    summary_df = series.reset_index().rename(columns={'index': 'Person'})
    potential = {p: 0 for p in assign_count.keys()}
    for p in potential.keys():
        potential[p] = sum(1 for d in service_dates if availability.get(p, {}).get(d, False))
    pot_df = pd.DataFrame({'Person': list(potential.keys()), 'AvailableYesDates': list(potential.values())})
    return summary_df.merge(pot_df, on='Person', how='left')

# ------------------------------
# UI
# ------------------------------
with st.sidebar:
    st.header('Upload Files (one month at a time)')
    people_file = st.file_uploader('Serving positions (Excel)', type=['xlsx', 'xls'])
    responses_file = st.file_uploader('Form responses (Excel)', type=['xlsx', 'xls'])
    run_btn = st.button('Generate Schedule')

if run_btn:
    if not people_file or not responses_file:
        st.error('Please upload both the serving positions and the form responses files.')
        st.stop()

    # Read files
    people_df = pd.read_excel(people_file)
    responses_df = pd.read_excel(responses_file)

    # Detect columns
    name_col_people = detect_name_column(people_df)
    name_col_resp = detect_name_column(responses_df)

    # Role columns as 0..5 numeric
    role_cols = [c for c in people_df.columns if c != name_col_people and is_priority_col(people_df[c])]
    if not role_cols:
        st.error('No role columns with priorities (0-5) detected in the serving positions file.')
        st.stop()

    # Build long form
    long_df = build_long_df(people_df, name_col_people, role_cols)
    if long_df.empty:
        st.error('No eligible assignments found (all priorities are 0 or missing).')
        st.stop()

    # Parse month and dates from headers
    try:
        year, month, date_map = parse_month_and_dates_from_headers(responses_df)
    except Exception as e:
        st.error(f'Could not parse month & dates from responses: {e}')
        st.stop()

    # Parse availability
    availability, service_dates = parse_availability(responses_df, name_col_resp, year, month, date_map)

    # Schedule (EXACT 2 only)
    schedule, assign_count = schedule_exact2(long_df, availability, service_dates, role_cols)

    st.success(f'Schedule generated for {date(service_dates[0].year, service_dates[0].month, 1):%B %Y}!')
    st.write('### Schedule')
    st.dataframe(schedule.reset_index().rename(columns={'index': 'Date'}))

    report_df = make_report(assign_count, availability, service_dates)
    st.write('### Assignment Summary')
    st.dataframe(report_df)

    # Downloads
    out_xlsx = io.BytesIO()
    with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
        schedule.reset_index().rename(columns={'index': 'Date'}).to_excel(writer, sheet_name='Schedule', index=False)
        report_df.to_excel(writer, sheet_name='AssignmentSummary', index=False)
        pd.DataFrame({'Roles': role_cols}).to_excel(writer, sheet_name='Roles', index=False)
    out_xlsx.seek(0)

    out_csv = io.StringIO()
    schedule.reset_index().rename(columns={'index': 'Date'}).to_csv(out_csv, index=False)
    out_csv.seek(0)

    st.download_button('Download Excel (.xlsx)', data=out_xlsx, file_name=f'uKids_schedule_{year}_{month:02d}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.download_button('Download CSV (.csv)', data=out_csv.getvalue(), file_name=f'uKids_schedule_{year}_{month:02d}.csv', mime='text/csv')

    # Diagnostics
    under = [p for p, c in assign_count.items() if c < 2]
    exact = [p for p, c in assign_count.items() if c == 2]
    st.write('### Diagnostics')
    st.write(f'People with < 2 assignments: {len(under)}')
    st.write(f'People with = 2 assignments: {len(exact)}')
else:
    st.info('Upload the two Excel files for a single month and click **Generate Schedule**. This app detects the month automatically from headers like "Are you available 7 September?".')









