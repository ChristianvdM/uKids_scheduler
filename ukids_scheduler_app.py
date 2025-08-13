import pandas as pd
import streamlit as st
import random
import datetime
from collections import defaultdict
from openpyxl import Workbook

# --- Upload section ---
st.title("uKids Scheduler - CSV Version")

availability_file = st.file_uploader("Upload availability CSV", type=["csv"])
positions_file = st.file_uploader("Upload positions CSV", type=["csv"])

if not availability_file or not positions_file:
    st.stop()

# --- Load inputs ---
availability_df = pd.read_csv(availability_file)
positions_df = pd.read_csv(positions_file)

# --- Parse availability ---
name_col_avail = [c for c in availability_df.columns if "name" in c.lower()][0]
date_cols = [c for c in availability_df.columns if "september" in c.lower()]
service_dates = date_cols

availability = {}
for _, row in availability_df.iterrows():
    name = str(row[name_col_avail]).strip()
    availability[name] = {date: str(row[date]).strip().lower() == "yes" for date in date_cols}

# --- Parse eligibility ---
name_col_positions = [c for c in positions_df.columns if "name" in c.lower()][0]
eligibility = defaultdict(dict)
for _, row in positions_df.iterrows():
    person = str(row[name_col_positions]).strip()
    if person not in availability:
        continue
    for role, val in row.items():
        if role == name_col_positions:
            continue
        try:
            level = int(val)
        except:
            continue
        if level != 0:
            eligibility[person][role.strip()] = level

# --- Define roles & capacity ---
roles = list({r for p in eligibility.values() for r in p})
role_capacity = {
    "Age 1-2": 2,
    "Age 3": 2,
    "Age 4": 2,
    "Age 5": 2,
    "Age 6": 2,
    "Age 7": 2,
    "Age 8": 2,
    "Age 9": 2,
    "Age 10": 2,
    "Age 11": 2,
    "Special needs": 2,
}

# Fill missing roles with default capacity = 2
for r in roles:
    if r not in role_capacity:
        role_capacity[r] = 2

# --- Assignment logic ---
def make_schedule():
    count = defaultdict(int)
    assignments = defaultdict(lambda: defaultdict(list))

    for date in service_dates:
        for role in roles:
            needed = role_capacity[role]
            eligible = [p for p in eligibility if role in eligibility[p] and availability[p].get(date, False) and count[p] < 2]
            random.shuffle(eligible)
            eligible.sort(key=lambda x: count[x])
            for p in eligible[:needed]:
                assignments[date][role].append(p)
                count[p] += 1

    return assignments, count

assignments, count = make_schedule()

# --- Format output ---
def generate_schedule_dataframe(assignments):
    index = sorted(roles)
    df = pd.DataFrame(index=index, columns=service_dates)
    for date in service_dates:
        for role in index:
            people = assignments[date].get(role, [])
            df.at[role, date] = ", ".join(people)
    return df

schedule_df = generate_schedule_dataframe(assignments)

# --- Display ---
st.success("✅ Schedule generated!")
st.dataframe(schedule_df)

# --- Stats ---
st.markdown("### Assignment Stats")
st.dataframe(pd.DataFrame.from_dict(count, orient='index', columns=["Times Assigned"]))

# --- Download ---
def to_excel(schedule_df):
    wb = Workbook()
    ws = wb.active
    ws.title = "Schedule"

    # Header
    ws.append(["Role"] + service_dates)
    for role in schedule_df.index:
        ws.append([role] + [schedule_df.at[role, date] or "" for date in service_dates])

    # Stats sheet
    stats = wb.create_sheet("Stats")
    stats.append(["Person", "Times Assigned"])
    for person, c in count.items():
        stats.append([person, c])

    from io import BytesIO
    bio = BytesIO()
    wb.save(bio)
    bio.seek(0)
    return bio

excel_data = to_excel(schedule_df)
st.download_button("📥 Download Schedule as Excel", excel_data, "ukids_schedule.xlsx")
