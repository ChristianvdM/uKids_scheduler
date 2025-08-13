import pandas as pd
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.utils import get_column_letter

# === Configurable section ===

# Role capacities (excluding leaders)
role_capacity = {
    'age 1 volunteers': 5,
    'age 1 nappies': 1,
    'age 1 bags girls': 1,
    'age 1 bags boys': 1,
    'age 2 volunteers': 4,
    'age 2 nappies': 1,
    'age 2 bags girls': 1,
    'age 2 bags boys': 1,
    'age 3 volunteers': 4,
    'age 3 bags': 1,
    'age 4 volunteers': 4,
    'age 5 volunteers': 3,
    'age 6 volunteers': 3,
    'age 7 volunteers': 2,
    'age 8 volunteers': 2,
    'age 9 volunteers': 2,  # x2 as specified
    'age 10 volunteers': 1,
    'age 11 volunteers': 1,
    'special needs volunteers': 2,
    'leaders': 1
}

max_assignments = 2

# === Load data ===

availability_df = pd.read_csv("Untitled form (Responses).csv")
positions_df = pd.read_csv("uKids serving positions (4).csv")

# === Clean and parse ===

# Standardize volunteer names
availability_df.iloc[:, 0] = availability_df.iloc[:, 0].astype(str).str.strip()
positions_df.iloc[:, 0] = positions_df.iloc[:, 0].astype(str).str.strip()

# Extract dates from columns
date_columns = availability_df.columns[1:]

# Convert availability to a dict of dicts: availability[name][date] = True/False
availability = {}
for _, row in availability_df.iterrows():
    name = row.iloc[0]
    availability[name] = {
        date: str(row[date]).strip().lower() == "yes"
        for date in date_columns
    }

# Convert positions to a nested dict: eligibility[name][role] = skill level
eligibility = {}
for _, row in positions_df.iterrows():
    name = row.iloc[0]
    if name not in availability:
        continue  # skip if not in both files
    eligibility[name] = {}
    for role, value in row.iloc[1:].items():
        try:
            level = int(value)
        except:
            level = 0
        eligibility[name][role.strip()] = level

# === Build assignment ===

assignments = defaultdict(list)  # (date, role) -> list of names
assignment_count = defaultdict(int)

for date in date_columns:
    for role, needed in role_capacity.items():
        # Get eligible people
        possible = [
            p for p in eligibility
            if availability[p].get(date, False)
            and role in eligibility[p]
            and eligibility[p][role] > 0
            and assignment_count[p] < max_assignments
            and p not in assignments[(date, role)]
        ]
        assigned = 0
        for p in sorted(possible, key=lambda x: assignment_count[x]):
            if assigned >= needed:
                break
            assignments[(date, role)].append(p)
            assignment_count[p] += 1
            assigned += 1

# === Write Excel ===

wb = Workbook()
ws = wb.active
ws.title = "September 2025"

# Create schedule layout
dates = list(date_columns)
roles = list(role_capacity.keys())

# Headers
for col, date in enumerate(dates, start=2):
    ws.cell(row=1, column=col, value=date)

for row, role in enumerate(roles, start=2):
    ws.cell(row=row, column=1, value=role)

# Fill assignments
for row, role in enumerate(roles, start=2):
    for col, date in enumerate(dates, start=2):
        key = (date, role)
        people = assignments.get(key, [])
        ws.cell(row=row, column=col, value=", ".join(people))

# Add stats sheet
ws2 = wb.create_sheet("Assignment Count")
ws2.append(["Name", "Assigned"])
for name, count in sorted(assignment_count.items(), key=lambda x: x[1], reverse=True):
    ws2.append([name, count])

# Save output
output_path = "/mnt/data/ukids_september_schedule_cleaned.xlsx"
wb.save(output_path)
print(f"✅ Schedule saved to {output_path}")
