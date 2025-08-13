import streamlit as st
import pandas as pd
import datetime
from collections import defaultdict
from io import BytesIO
from openpyxl import Workbook

st.set_page_config(layout="wide")
st.title("🗓️ uKids Scheduler – September 2025")

# --- File Upload ---
availability_file = st.file_uploader("Upload availability CSV", type=["csv"])
positions_file = st.file_uploader("Upload positions CSV", type=["csv"])

if availability_file and positions_file:
    availability_df = pd.read_csv(availability_file)
    positions_df = pd.read_csv(positions_file)

    # --- Normalize Names and Dates ---
    name_col = "What is your name AND surname?"
    date_cols = [col for col in availability_df.columns if "September" in col or "7 September" in col]

    service_dates = []
    for col in date_cols:
        try:
            day = int(col.split()[2])
            date = datetime.date(2025, 9, day)
            service_dates.append((col, date))
        except Exception:
            continue

    # --- Build Availability Dictionary ---
    availability = {}
    for _, row in availability_df.iterrows():
        name = str(row[name_col]).strip()
        availability[name] = {}
        for col, date in service_dates:
            availability[name][date] = str(row[col]).strip().lower() == "yes"

    # --- Build Eligibility Dictionary ---
    eligibility = defaultdict(dict)
    roles = set()
    for _, row in positions_df.iterrows():
        person = str(row["Name"]).strip()
        role = str(row["Role"]).strip()
        level = int(row["Level"])
        if level > 0:
            eligibility[person][role] = level
            roles.add(role)

    # --- Prepare Assignments ---
    assignments = defaultdict(list)
    count = defaultdict(int)

    for date_label, date in service_dates:
        for role in roles:
            needed = int(positions_df[(positions_df["Role"] == role)]["Capacity"].max())
            possible = [
                p for p in eligibility
                if availability.get(p, {}).get(date, False)
                and role in eligibility[p]
                and count[p] < 2
                and len(assignments[(date, role)]) < needed
            ]

            assigned = 0
            for p in sorted(possible, key=lambda x: count[x]):
                if assigned >= needed:
                    break
                assignments[(date, role)].append(p)
                count[p] += 1
                assigned += 1

    # --- Format for Excel ---
    excel_output = BytesIO()
    wb = Workbook()
    ws = wb.active
    ws.title = "September 2025"

    sorted_roles = sorted(list(roles))
    sorted_dates = sorted([date for _, date in service_dates])

    # Headers
    ws.cell(row=1, column=1).value = "Role"
    for j, date in enumerate(sorted_dates):
        ws.cell(row=1, column=j + 2).value = date.strftime("%-d %b")

    # Data
    for i, role in enumerate(sorted_roles):
        ws.cell(row=i + 2, column=1).value = role
        for j, date in enumerate(sorted_dates):
            people = assignments.get((date, role), [])
            ws.cell(row=i + 2, column=j + 2).value = ", ".join(people)

    # Stats sheet
    stats_ws = wb.create_sheet("Stats")
    stats_ws.cell(row=1, column=1).value = "Name"
    stats_ws.cell(row=1, column=2).value = "Assignments"
    for i, (name, cnt) in enumerate(sorted(count.items(), key=lambda x: -x[1]), start=2):
        stats_ws.cell(row=i, column=1).value = name
        stats_ws.cell(row=i, column=2).value = cnt

    wb.save(excel_output)
    excel_output.seek(0)

    # --- Download Link ---
    st.success("✅ Schedule generated!")
    st.download_button(
        label="📥 Download Schedule (Excel)",
        data=excel_output,
        file_name="ukids_schedule_september_2025.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
