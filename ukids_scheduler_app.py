import streamlit as st
import pandas as pd
from collections import defaultdict

st.set_page_config(layout="wide", page_title="uKids Scheduler")
st.title("📅 uKids Schedule Generator")

st.markdown("Upload the two required files: availability (form) and serving positions.")

availability_file = st.file_uploader("Upload Availability Form", type=["xlsx"])
positions_file = st.file_uploader("Upload Serving Positions File", type=["xlsx"])

if availability_file and positions_file:
    availability_df = pd.read_excel(availability_file)
    positions_df = pd.read_excel(positions_file)

    # Parse dates from header
    service_dates = [col for col in availability_df.columns if "September" in col]

    # Standardize names
    availability_df["Name"] = availability_df.iloc[:, 1].str.strip()
    availability = {
        row["Name"]: {
            date: str(row[date]).strip().lower() == "yes" for date in service_dates
        } for _, row in availability_df.iterrows()
    }

    # Transform positions into long format
    positions_df = positions_df.rename(columns={positions_df.columns[0]: "Name"})
    long_df = positions_df.melt(id_vars=["Name"], var_name="Role", value_name="Skill")
    long_df.dropna(inplace=True)
    long_df["Skill"] = pd.to_numeric(long_df["Skill"], errors="coerce").fillna(0).astype(int)
    long_df["Name"] = long_df["Name"].str.strip()

    # Keep only people present in BOTH sources
    common_people = set(long_df["Name"]).intersection(set(availability.keys()))
    long_df = long_df[long_df["Name"].isin(common_people)]
    availability = {k: v for k, v in availability.items() if k in common_people}

    # Build eligibility map
    eligibility = defaultdict(dict)
    for _, row in long_df.iterrows():
        if row["Skill"] > 0:
            eligibility[row["Name"]][row["Role"]] = row["Skill"]

    people = list(eligibility.keys())

    # Build list of unique roles
    roles = long_df["Role"].unique()
    role_capacity = {role: 1 for role in roles}  # Default 1 per role

    # Override known group roles with capacity 2
    for r in roles:
        if any(x in r.lower() for x in ["classroom", "bags", "nappies", "volunteers"]):
            role_capacity[r] = 2

    def get_candidates(role, date, people, eligibility, availability, count, max_count, assigned_today):
        out = []
        for p in people:
            if count.get(p, 0) >= max_count:
                continue
            if p in assigned_today:
                continue
            if not availability.get(p, {}).get(date, False):
                continue
            pr_map = eligibility.get(p, {})
            if role not in pr_map:
                continue
            out.append((p, pr_map[role]))
        return out

    def make_assignment():
        assignments = defaultdict(lambda: defaultdict(str))
        count = defaultdict(int)

        for pass_level in [2, 3]:  # First pass = max 2 per person, second = allow 3
            for date in service_dates:
                assigned_today = set()
                for role in roles:
                    remaining = role_capacity[role] - sum(1 for x in assignments[date].values() if x == role)
                    while remaining > 0:
                        candidates = get_candidates(role, date, people, eligibility, availability, count, pass_level, assigned_today)
                        if not candidates:
                            break
                        candidates.sort(key=lambda x: (x[1], count[x[0]]))
                        selected = candidates[0][0]
                        key = f"{role} ({role_capacity[role]})"
                        # Place in first available spot
                        for i in range(role_capacity[role]):
                            slot = f"{key}-{i+1}"
                            if slot not in assignments[date]:
                                assignments[date][slot] = selected
                                break
                        count[selected] += 1
                        assigned_today.add(selected)
                        remaining -= 1

        return assignments, count

    def format_schedule(assignments, service_dates):
        all_roles = sorted(set(k for date in assignments for k in assignments[date]))
        df = pd.DataFrame(index=all_roles, columns=service_dates)
        for date in service_dates:
            for role in all_roles:
                df.at[role, date] = assignments[date].get(role, "")
        return df.reset_index().rename(columns={"index": "Role"})

    assignments, count = make_assignment()
    schedule_df = format_schedule(assignments, service_dates)

    st.success("✅ Schedule generated!")
    st.dataframe(schedule_df, use_container_width=True)

    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(schedule_df)
    st.download_button("📥 Download Schedule CSV", csv, "ukids_schedule.csv", "text/csv")
