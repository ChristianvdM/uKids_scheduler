import pandas as pd
import streamlit as st
from io import StringIO
from openpyxl import Workbook
from datetime import datetime
import base64

st.set_page_config(layout="wide")
st.title("uKids September Scheduler")

# Upload CSV files
availability_file = st.file_uploader("Upload availability CSV", type="csv", key="avail")
positions_file = st.file_uploader("Upload serving positions CSV", type="csv", key="roles")

if not availability_file or not positions_file:
    st.warning("Please upload both availability and serving positions CSV files.")
    st.stop()

# Read files
availability_df = pd.read_csv(availability_file)
positions_df = pd.read_csv(positions_file)

# Extract service dates
service_dates = [col for col in availability_df.columns if "September" in col]

# Standardize names
availability_df["Name"] = availability_df.iloc[:, 1].str.strip()

# Convert availability to dict
availability = {}
for _, row in availability_df.iterrows():
    name = row["Name"]
    availability[name] = {}
    for date in service_dates:
        availability[name][date] = str(row[date]).strip().lower() == "yes"

# Parse positions/preferences
positions = []
eligibility = {}
for _, row in positions_df.iterrows():
    person = row["Name"].strip()
    if person not in availability:
        continue
    eligibility[person] = {}
    for role in positions_df.columns[1:]:
        try:
            pr = int(row[role])
        except:
            pr = 5
        eligibility[person][role] = pr
        if role not in positions:
            positions.append(role)

# Count of assignments
count = {name: 0 for name in availability}
assignments = {(date, role): [] for date in service_dates for role in positions}

# Capacity rules
capacity = {
    "Age 3 leader": 1,
    "Age 3 volunteers": 2,
    "Age 4 leader": 1,
    "Age 4 volunteers": 2,
    "Age 5 leader": 1,
    "Age 5 volunteers": 2,
    "Grade R leader": 1,
    "Grade R volunteers": 2,
    "Grade 1 leader": 1,
    "Grade 1 volunteers": 2,
    "Age 6-8 worship": 2,
    "Grade 2-3 leader": 1,
    "Grade 2-3 volunteers": 2,
    "Grade 4-5 leader": 1,
    "Grade 4-5 volunteers": 2,
    "Age 11 leader": 1,
    "Age 11 volunteers": 2,
    "Special needs volunteers": 2,
}

# Assign roles
for date in service_dates:
    for role in positions:
        if role not in capacity:
            continue
        needed = capacity[role]
        possible = [p for p in eligibility if availability[p].get(date, False)
                    and role in eligibility[p]
                    and eligibility[p][role] > 0
                    and count[p] < 2
                    and len(assignments[(date, role)]) < needed]
        assigned = 0
        for p in sorted(possible, key=lambda x: (count[x])):
            if assigned >= needed:
                break
            assignments[(date, role)].append(p)
            count[p] += 1
            assigned += 1

# Output DataFrame
schedule = pd.DataFrame(index=positions, columns=service_dates)
for date in service_dates:
    for role in positions:
        people = assignments.get((date, role), [])
        schedule.at[role, date] = ", ".join(people)

# Display
st.dataframe(schedule.fillna(""))

# Stats
st.markdown("### Summary Statistics")
assigned_df = pd.DataFrame.from_dict(count, orient="index", columns=["Times Assigned"])
st.dataframe(assigned_df.sort_values("Times Assigned", ascending=False))

# Export
def to_excel(schedule_df):
    output = StringIO()
    writer = pd.ExcelWriter("schedule_output.xlsx", engine='openpyxl')
    schedule_df.to_excel(writer, sheet_name="Schedule")
    assigned_df.to_excel(writer, sheet_name="Stats")
    writer.close()
    return "schedule_output.xlsx"

if st.button("📥 Download Schedule"):
    filename = to_excel(schedule)
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="schedule_output.xlsx">Click here to download your schedule</a>'
    st.markdown(href, unsafe_allow_html=True)
