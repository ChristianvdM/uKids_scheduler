
# ukids_scheduler_app.py
# Streamlit app for uKids Scheduling
import io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
from typing import Dict, List, Tuple
import base64
import re
from io import BytesIO
from collections import defaultdict

st.set_page_config(page_title='uKids Scheduler', layout='wide')
st.title('uKids Scheduler')

# Logo
try:
    with open("image(1).png", "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
            <div style='text-align: center;'>
                <img src='data:image/png;base64,{encoded}' width='600'>
            </div>
        """, unsafe_allow_html=True)
except:
    st.warning("Logo not found")

# Constants
MONTH_ALIASES = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12
}
YES_SET = {"yes", "y", "true", "available"}
MOST_PEOPLE_RATIO = 0.80

# Upload files
people_file = st.file_uploader("Upload Serving Positions (.xlsx)", type="xlsx")
responses_file = st.file_uploader("Upload Form Responses (.xlsx)", type="xlsx")
submit_btn = st.button("Generate Schedule")

# Helpers
def normalize(s): return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()
def detect_name_col(df): return next((c for c in df.columns if "name" in c.lower()), df.columns[0])
def is_priority_col(series): return pd.to_numeric(series, errors="coerce").between(0, 5).all()

def parse_dates(responses_df):
    date_cols = [c for c in responses_df.columns if "available" in c.lower()]
    result = {}
    for col in date_cols:
        m = re.search(r"(\d{1,2})\s*([a-zA-Z]+)", col)
        if m:
            day, mon = int(m.group(1)), MONTH_ALIASES[m.group(2).lower()[:3]]
            year = pd.to_datetime(responses_df["Timestamp"]).dt.year.mode()[0]
            result[col] = pd.Timestamp(year=year, month=mon, day=day)
    return result

def parse_availability(df, name_col, date_map):
    availability = defaultdict(dict)
    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        for col, dt in date_map.items():
            availability[name][dt] = str(row[col]).strip().lower() in YES_SET
    return availability

def build_long_df(people_df, name_col):
    long_rows = []
    for _, row in people_df.iterrows():
        person = str(row[name_col]).strip()
        for col in people_df.columns:
            if col == name_col: continue
            if is_priority_col(people_df[col]):
                level = int(pd.to_numeric(row[col], errors="coerce") or 0)
                if level > 0:
                    long_rows.append({"person": person, "role": col, "priority": level})
    return pd.DataFrame(long_rows)

def get_people_with_few_dates(availability, min_dates=2):
    return [name for name, dates in availability.items()
            if sum(dates.values()) < min_dates]

def make_assignment(long_df, availability, service_dates):
    rng = np.random.default_rng(42)
    people = long_df["person"].unique()
    roles = long_df["role"].unique()
    role_capacity = defaultdict(lambda: 1)
    for role in roles:
        if "classroom" in role: role_capacity[role] = 5
        elif "leader" in role: role_capacity[role] = 1
        else: role_capacity[role] = 1

    count = defaultdict(int)
    assignments = {(role, date): [] for role in roles for date in service_dates}
    eligibility = defaultdict(dict)
    for _, row in long_df.iterrows():
        eligibility[row["person"]][row["role"]] = row["priority"]

    def get_candidates(role, date, max_count):
        return [(p, eligibility[p][role])
                for p in people
                if count[p] < max_count and
                availability[p].get(date, False) and
                role in eligibility[p]]

    for pass_level in [2, 3]:
        for date in service_dates:
            assigned_today = set()
            for role in roles:
                remaining = role_capacity[role] - len(assignments[(role, date)])
                while remaining > 0:
                    candidates = get_candidates(role, date, pass_level)
                    if not candidates: break
                    candidates.sort(key=lambda x: (x[1], count[x[0]]))
                    selected = candidates[0][0]
                    assignments[(role, date)].append(selected)
                    count[selected] += 1
                    assigned_today.add(selected)
                    remaining -= 1

    return assignments, count

def format_schedule(assignments, service_dates, role_order=None):
    roles = sorted({r for (r, _) in assignments}) if role_order is None else role_order
    df = pd.DataFrame(index=roles, columns=service_dates)
    for (role, date), names in assignments.items():
        df.loc[role, date] = ", ".join(names)
    df = df.fillna("")
    df.columns = [d.strftime("%d %b") for d in df.columns]
    return df

if submit_btn and people_file and responses_file:
    people_df = pd.read_excel(people_file)
    responses_df = pd.read_excel(responses_file)

    name_col = detect_name_col(people_df)
    response_name_col = detect_name_col(responses_df)
    date_map = parse_dates(responses_df)
    service_dates = sorted(date_map.values())

    long_df = build_long_df(people_df, name_col)
    availability = parse_availability(responses_df, response_name_col, date_map)

    excluded_names = get_people_with_few_dates(availability, min_dates=2)
    availability = {k: v for k, v in availability.items() if k not in excluded_names}

    assignments, count = make_assignment(long_df, availability, service_dates)
    schedule_df = format_schedule(assignments, service_dates)

    st.success("✅ Schedule generated!")
    st.dataframe(schedule_df, use_container_width=True)

    # Excel export
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        schedule_df.to_excel(writer, sheet_name="August 2025")
        pd.DataFrame({"Name": excluded_names}).to_excel(writer, sheet_name="Unavailable (<2 dates)", index=False)
        count_df = pd.DataFrame(list(count.items()), columns=["Name", "Assignments"])
        count_df.to_excel(writer, sheet_name="Assignments", index=False)
        worksheet = writer.sheets["August 2025"]
        for i, width in enumerate([30]*len(schedule_df.columns)):
            worksheet.set_column(i+1, i+1, width)

    st.download_button("Download Excel", output.getvalue(), file_name="uKids_schedule.xlsx")
