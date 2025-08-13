import pandas as pd
import streamlit as st
from io import BytesIO
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Alignment

# --- File Upload ---
availability_file = st.file_uploader("Upload availability CSV", type="csv")
positions_file = st.file_uploader("Upload serving positions CSV", type="csv")

if not availability_file or not positions_file:
    st.stop()

# --- Read Data ---
availability_df = pd.read_csv(availability_file)
positions_df = pd.read_csv(positions_file)

# --- Parse availability ---
name_col = [col for col in availability_df.columns if "name" in col.lower()][0]
date_cols = [col for col in availability_df.columns if "september" in col.lower()]

availability = {}
for _, row in availability_df.iterrows():
    name = row[name_col].strip()
    availability[name] = {col: row[col].strip().lower() == "yes" for col in date_cols}

# --- Parse positions ---
roles = positions_df.columns[1:]  # first col is name
eligibility = {}
for _, row in positions_df.iterrows():
    name = row[0].strip()
    if name not in availability:
        continue
    eligibility[name] = {}
    for role in roles:
        val = row[role]
        try:
            val = int(val)
        except:
            val = 0
        eligibility[name][role] = val

# --- Assignment Logic ---
service_dates = date_cols
assignments = {(role, date): [] for role in roles for date in service_dates}
count = {name: 0 for name in eligibility}
max_assignments = 2

for date in service_dates:
    for role in roles:
        capacity = positions_df[role].max()
        while len(assignments[(role, date)]) < capacity:
            candidates = [
                p for p in eligibility
                if eligibility[p][role] > 0 and
                   availability[p][date] and
                   count[p] < max_assignments and
                   p not in assignments[(role, date)]
            ]
            if not candidates:
                break
            # Choose lowest count person
            candidates.sort(key=lambda x: count[x])
            chosen = candidates[0]
            assignments[(role, date)].append(chosen)
            count[chosen] += 1

# --- Output Table ---
schedule_df = pd.DataFrame(index=roles, columns=service_dates)
for role in roles:
    for date in service_dates:
        schedule_df.loc[role, date] = ", ".join(assignments[(role, date)])

# --- Stats Table ---
stats_df = pd.DataFrame.from_dict(count, orient='index', columns=['Times Assigned'])
stats_df.index.name = 'Name'

# --- Excel Export ---
buffer = BytesIO()
wb = Workbook()
ws = wb.active
ws.title = "Schedule"

for r in dataframe_to_rows(schedule_df, index=True, header=True):
    ws.append(r)
for col in ws.columns:
    for cell in col:
        cell.alignment = Alignment(wrap_text=True, vertical="top")
        cell.column_letter = cell.column_letter
    ws.column_dimensions[cell.column_letter].width = 25

ws2 = wb.create_sheet("Stats")
for r in dataframe_to_rows(stats_df.reset_index(), index=False, header=True):
    ws2.append(r)

wb.save(buffer)
st.download_button("📥 Download Schedule Excel", data=buffer.getvalue(), file_name="uKids Schedule.xlsx")
