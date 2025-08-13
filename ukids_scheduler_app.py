import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from openpyxl import Workbook

# Load CSV inputs
availability_df = pd.read_csv("Untitled form (Responses) - Form Responses 1.csv")
positions_df = pd.read_csv("uKids serving positions.xlsx - Pretoria.csv")

# Standardize column headers
availability_df.columns = availability_df.columns.str.strip()
positions_df.columns = positions_df.columns.str.strip()

# Extract dates from availability columns
date_cols = [col for col in availability_df.columns if col.startswith("Are you available")]
service_dates = [col.split("Are you available")[-1].strip(" ?") for col in date_cols]

# Map names to availability by date
availability = defaultdict(dict)
for _, row in availability_df.iterrows():
    name = row["What is your name AND surname?"].strip()
    for col in date_cols:
        date = col.split("Are you available")[-1].strip(" ?")
        availability[name][date] = str(row[col]).strip().lower() == "yes"

# Map eligibility per person per role (remove 0's)
eligibility = defaultdict(dict)
for _, row in positions_df.iterrows():
    name = row.iloc[0].strip()
    for role, val in row.iloc[1:].items():
        try:
            score = int(val)
            if score > 0:
                eligibility[name][role.strip()] = score
        except:
            continue

# Build long dataframe of eligible people for each role
data = []
for person, roles in eligibility.items():
    for role, score in roles.items():
        data.append({"person": person, "role": role, "priority": score})
long_df = pd.DataFrame(data)

# Get full unique list of roles
roles = sorted(long_df['role'].unique())

# Assignments per date, per role
assignments = defaultdict(lambda: defaultdict(str))
assignment_counts = Counter()

# Calculate capacity: assume 1 per role by default, can customize if needed
role_capacity = {role: 1 for role in roles}

# Max 2 assignments per person
max_assignments = 2

for date in service_dates:
    for role in roles:
        assigned = 0
        candidates = [p for p in long_df[long_df['role'] == role]['person'].unique()
                      if availability[p].get(date, False)
                      and assignment_counts[p] < max_assignments]
        np.random.shuffle(candidates)  # randomness to distribute load
        for person in candidates:
            if assigned >= role_capacity[role]:
                break
            if person not in assignments[date].values():
                assignments[date][role] = person
                assignment_counts[person] += 1
                assigned += 1

# Build schedule DataFrame
schedule_df = pd.DataFrame(index=roles, columns=service_dates)
for date in service_dates:
    for role in roles:
        schedule_df.loc[role, date] = assignments[date].get(role, "")

# Build stats sheet
stats_df = pd.DataFrame.from_dict(assignment_counts, orient='index', columns=['Assignments'])
stats_df = stats_df.sort_values(by="Assignments", ascending=False)

# Save to Excel
out_path = "uKids_September_Schedule.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    schedule_df.to_excel(writer, sheet_name="Schedule")
    stats_df.to_excel(writer, sheet_name="Stats")

print(f"✅ Schedule saved to {out_path}")
