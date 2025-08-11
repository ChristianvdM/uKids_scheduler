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
# Config
# ------------------------------
MOST_PEOPLE_RATIO = 0.80   # if >=80% of people reached 2, allow a 3rd assignment to fill gaps
RANDOM_SEED = 123          # deterministic tie-breaks


# ------------------------------
# Helpers
# ------------------------------
MONTH_ALIASES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

YES_SET = {"yes", "y", "true", "available"}

DEFAULT_CAPACITIES = [
    {"Role": "Age 1 volunteers", "Capacity": 5},
    {"Role": "Age 1 nappies", "Capacity": 1},
    {"Role": "Age 1 bags girls", "Capacity": 1},
    {"Role": "Age 1 bags boys", "Capacity": 1},
    {"Role": "Age 2 volunteers", "Capacity": 4},
    {"Role": "Age 2 nappies", "Capacity": 1},
    {"Role": "Age 2 bags girls", "Capacity": 1},
    {"Role": "Age 2 bags boys", "Capacity": 1},
    {"Role": "Age 3 volunteers", "Capacity": 4},
    {"Role": "Age 3 bags", "Capacity": 1},
    {"Role": "Age 4 volunteers", "Capacity": 4},
    {"Role": "Age 5 volunteers", "Capacity": 3},
    {"Role": "Age 6 volunteers", "Capacity": 3},
    {"Role": "Age 7 volunteers", "Capacity": 2},
    {"Role": "Age 8 volunteers", "Capacity": 2},
    {"Role": "Age 9 volunteers", "Capacity": 2},
    {"Role": "Age 10 volunteers", "Capacity": 1},
    {"Role": "Age 11 volunteers", "Capacity": 1},
    {"Role": "Special needs volunteers", "Capacity": 2},
]

def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()

def detect_name_column(df: pd.DataFrame) -> str:
    prefs = ["Name", "Full name", "Full names", "What is your name AND surname?"]
    for pref in prefs:
        for c in df.columns:
            if isinstance(c, str) and c.strip().lower() == pref.strip().lower():
                return c
    candidates = [c for c in df.columns if isinstance(c, str) and "name" in c.lower()]
    return candidates[0] if candidates else df.columns[0]

def is_priority_col(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return False
    return (vals.min() >= 0) and (vals.max() <= 5)

def build_long_df(people_df: pd.DataFrame, name_col: str, role_cols: List[str]) -> pd.DataFrame:
    records = []
    for _, r in people_df.iterrows():
        person = str(r[name_col]).strip()
        if not person or person.lower() == "nan":
            continue
        for role in role_cols:
            pr = pd.to_numeric(r[role], errors="coerce")
            if pd.isna(pr):
                continue
            pr = int(round(pr))
            if pr >= 1:  # only roles with priority 1..5 (0 is ineligible)
                records.append({"person": person, "role": role, "priority": pr})
    return pd.DataFrame(records)

def parse_month_and_dates_from_headers(responses_df: pd.DataFrame) -> Tuple[int, int, Dict[str, pd.Timestamp]]:
    avail_cols = [c for c in responses_df.columns if isinstance(c, str) and c.strip().lower().startswith("are you available")]
    if not avail_cols:
        raise ValueError("No 'Are you available ...' columns found in the responses. Upload one month at a time.")
    col_info = []
    for c in avail_cols:
        c_low = c.lower()
        month_name = None
        for alias in MONTH_ALIASES.keys():
            if alias in c_low:
                month_name = alias
                break
        day_match = re.search(r"(\d{1,2})", c_low)
        day = int(day_match.group(1)) if day_match else None
        if month_name and day:
            col_info.append((c, MONTH_ALIASES[month_name], day))
    if not col_info:
        raise ValueError("Could not parse day/month from availability headers. Expected e.g. 'Are you available 7 September?'")
    month_set = {m for _, m, _ in col_info}
    if len(month_set) > 1:
        raise ValueError(f"Multiple months detected in availability headers: {sorted(month_set)}. Upload one month at a time.")
    month = month_set.pop()
    if "Timestamp" in responses_df.columns:
        ts_years = pd.to_datetime(responses_df["Timestamp"], errors="coerce").dt.year.dropna().astype(int)
        year = int(ts_years.mode().iloc[0]) if not ts_years.empty else date.today().year
    else:
        year = date.today().year
    date_map = {c: pd.Timestamp(datetime(year, month, d)).normalize() for c, month, d in col_info}
    return year, month, date_map

def parse_availability(responses_df: pd.DataFrame, name_col_resp: str, date_map: Dict[str, pd.Timestamp]):
    availability: Dict[str, Dict[pd.Timestamp, bool]] = {}
    for _, row in responses_df.iterrows():
        nm = str(row.get(name_col_resp, "")).strip()
        if not nm or nm.lower() == "nan":
            continue
        availability.setdefault(nm, {})
        for col, dt in date_map.items():
            ans = str(row.get(col, "")).strip().lower()
            is_yes = ans in YES_SET
            availability[nm][dt] = is_yes
    service_dates = sorted(set(date_map.values()))
    return availability, service_dates

def map_capacity_roles_to_sheet(cap_df: pd.DataFrame, people_role_cols: List[str]) -> Dict[str, List[str]]:
    sheet_norm = {normalize(c): c for c in people_role_cols}
    mapping: Dict[str, List[str]] = {}
    for _, row in cap_df.iterrows():
        cap_role = str(row["Role"]).strip()
        norm = normalize(cap_role)
        matches = []
        if norm in sheet_norm:
            matches = [sheet_norm[norm]]
        else:
            for k_norm, original in sheet_norm.items():
                if all(tok in k_norm for tok in norm.split() if tok not in {"volunteers", "volunteer"}):
                    matches.append(original)
        mapping[cap_role] = sorted(set(matches))
    return mapping

def schedule_with_capacities(long_df: pd.DataFrame,
                             availability: Dict[str, Dict[pd.Timestamp, bool]],
                             service_dates: List[pd.Timestamp],
                             capacity_df: pd.DataFrame,
                             role_map: Dict[str, List[str]]) -> Tuple[pd.DataFrame, Dict[str, int], pd.DataFrame, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    target = 2
    people = long_df["person"].unique()
    assign_count = {p: 0 for p in people}

    # eligibility by person (sheet-role -> priority)
    elig_by_person = {}
    for p in people:
        sub = long_df[long_df["person"] == p]
        elig_by_person[p] = {row["role"]: int(row["priority"]) for _, row in sub.iterrows()}

    role_names = list(capacity_df["Role"])
    schedule_cells = {(role, d): [] for role in role_names for d in service_dates}
    unfilled = []

    # -------- PASS 1: Fill up to exactly 2 per person --------
    for d in service_dates:
        assigned_today = set()
        for _, crow in capacity_df.iterrows():
            cap_role = str(crow["Role"]).strip()
            cap = int(crow["Capacity"])
            if cap <= 0:
                continue
            mapped_sheet_roles = role_map.get(cap_role, [])
            if not mapped_sheet_roles:
                unfilled.append({"Role": cap_role, "Date": d.strftime("%Y-%m-%d"), "MissingCount": cap})
                continue

            # Candidates: available, not assigned today, assign_count < 2, eligible for ANY mapped role
            def candidate_list(max_allowed: int):
                cands = []
                for p in people:
                    if assign_count[p] >= max_allowed:
                        continue
                    if p in assigned_today:
                        continue
                    if not availability.get(p, {}).get(d, False):
                        continue
                    elig_roles = [(rname, elig_by_person[p][rname]) for rname in mapped_sheet_roles if rname in elig_by_person[p]]
                    if not elig_roles:
                        continue
                    best_pr = min(pr for _, pr in elig_roles)  # 1=best, 5=worst
                    cands.append((p, best_pr))
                return cands

            remaining = cap
            candidates = candidate_list(max_allowed=2)
            while remaining > 0 and candidates:
                # prefer lower total assigned, then lower priority number (1 best)
                min_count = min(assign_count[p] for p, _ in candidates)
                pool = [(p, pr) for (p, pr) in candidates if assign_count[p] == min_count]
                pool.sort(key=lambda x: (x[1], rng.random()))
                chosen, pr = pool[0]
                schedule_cells[(cap_role, d)].append(chosen)
                assign_count[chosen] += 1
                assigned_today.add(chosen)
                remaining -= 1
                candidates = [(p, pr_) for (p, pr_) in candidates if p != chosen]

            if remaining > 0:
                unfilled.append({"Role": cap_role, "Date": d.strftime("%Y-%m-%d"), "MissingCount": remaining})

    ratio_with_two = (sum(1 for p in people if assign_count[p] >= 2) / max(1, len(people)))

    # -------- PASS 2 (optional): Allow a 3rd assignment if most people reached 2 --------
    if unfilled and ratio_with_two >= MOST_PEOPLE_RATIO:
        # Try to fill remaining using people at exactly 2 (cap at 3), still no double-booking per date
        # Recompute "remaining" per (role, date)
        remaining_map = {}
        for (role, d), names in schedule_cells.items():
            cap = int(capacity_df.loc[capacity_df["Role"] == role, "Capacity"].iloc[0])
            rem = max(0, cap - len(names))
            if rem > 0:
                remaining_map[(role, d)] = rem

        for d in service_dates:
            assigned_today = set(name for (role, dd), names in schedule_cells.items() if dd == d for name in names)
            for _, crow in capacity_df.iterrows():
                cap_role = str(crow["Role"]).strip()
                rem = remaining_map.get((cap_role, d), 0)
                if rem <= 0:
                    continue
                mapped_sheet_roles = role_map.get(cap_role, [])
                if not mapped_sheet_roles:
                    continue

                # Candidates: people at exactly 2, available, eligible, not assigned today
                candidates = []
                for p in people:
                    if assign_count[p] != 2:
                        continue
                    if p in assigned_today:
                        continue
                    if not availability.get(p, {}).get(d, False):
                        continue
                    elig_roles = [(rname, elig_by_person[p][rname]) for rname in mapped_sheet_roles if rname in elig_by_person[p]]
                    if not elig_roles:
                        continue
                    best_pr = min(pr for _, pr in elig_roles)  # 1=best
                    candidates.append((p, best_pr))

                while rem > 0 and candidates:
                    # among equal 2-count people, prefer lower priority number
                    candidates.sort(key=lambda x: (x[1], rng.random()))
                    chosen, pr = candidates[0]
                    schedule_cells[(cap_role, d)].append(chosen)
                    assign_count[chosen] += 1  # now 3
                    assigned_today.add(chosen)
                    rem -= 1
                    candidates = [(p, pr_) for (p, pr_) in candidates if p != chosen]
                remaining_map[(cap_role, d)] = rem

        # Rebuild unfilled list after pass 2
        unfilled = []
        for (role, d), names in schedule_cells.items():
            cap = int(capacity_df.loc[capacity_df["Role"] == role, "Capacity"].iloc[0])
            rem = max(0, cap - len(names))
            if rem > 0:
                unfilled.append({"Role": role, "Date": d.strftime("%Y-%m-%d"), "MissingCount": rem})

    # Build display
    disp = pd.DataFrame(index=role_names, columns=service_dates)
    for (role, d), names in schedule_cells.items():
        disp.loc[role, d] = ", ".join(names)
    disp = disp.fillna("")
    disp.columns = [d.strftime("%Y-%m-%d") for d in disp.columns]
    unfilled_df = pd.DataFrame(unfilled)

    return disp, assign_count, unfilled_df, ratio_with_two


# ------------------------------
# UI (no date/month controls)
# ------------------------------
with st.sidebar:
    st.header("Upload Files (single month only)")
    people_file = st.file_uploader("Serving positions (Excel)", type=["xlsx", "xls"])
    responses_file = st.file_uploader("Form responses (Excel)", type=["xlsx", "xls"])
    st.markdown("Capacities per date (edit if needed):")
    cap_df = st.data_editor(pd.DataFrame(DEFAULT_CAPACITIES), num_rows="dynamic")
    run_btn = st.button("Generate Schedule")

if run_btn:
    if not people_file or not responses_file:
        st.error("Please upload both the serving positions and the form responses files.")
        st.stop()

    people_df = pd.read_excel(people_file)
    responses_df = pd.read_excel(responses_file)

    name_col_people = detect_name_column(people_df)
    name_col_resp = detect_name_column(responses_df)

    role_cols = [c for c in people_df.columns if c != name_col_people and is_priority_col(people_df[c])]
    if not role_cols:
        st.error("No role columns with priorities (0-5) detected in the serving positions file.")
        st.stop()

    long_df = build_long_df(people_df, name_col_people, role_cols)
    if long_df.empty:
        st.error("No eligible assignments found (all priorities are 0 or missing).")
        st.stop()

    try:
        year, month, date_map = parse_month_and_dates_from_headers(responses_df)
    except Exception as e:
        st.error(f"Could not parse month & dates from responses: {e}")
        st.stop()

    availability, service_dates = parse_availability(responses_df, name_col_resp, date_map)

    role_map = map_capacity_roles_to_sheet(cap_df, role_cols)

    schedule_display, assign_count, unfilled_df, ratio_with_two = schedule_with_capacities(
        long_df, availability, service_dates, cap_df, role_map
    )

    st.success(f"Schedule generated for {date(service_dates[0].year, service_dates[0].month, 1):%B %Y}!")
    st.write("### Schedule (positions as rows, dates as columns)")
    st.dataframe(schedule_display)

    # Role mapping diagnostics
    st.write("### Role Mapping (capacities -> sheet columns)")
    mapping_rows = []
    for cap_role, mapped in role_map.items():
        mapping_rows.append({"Capacity Role": cap_role, "Matched Sheet Roles": ", ".join(mapped) if mapped else "(no match)"})
    st.dataframe(pd.DataFrame(mapping_rows))

    # Assignment summary
    st.write("### Assignment Summary")
    series = pd.Series(assign_count, name="AssignedCount").sort_values(ascending=False)
    report_df = series.reset_index().rename(columns={"index": "Person"})
    st.dataframe(report_df)

    # Coverage note
    st.info(f"{ratio_with_two:.0%} of people reached 2 assignments. "
            f"{'A 3rd-pass fill was applied.' if ratio_with_two >= MOST_PEOPLE_RATIO else 'No 3rd-pass fill applied (threshold 80%).'}")

    if not unfilled_df.empty:
        st.warning("Some slots could not be filled even after the 3rd-pass rule. See below:")
        st.dataframe(unfilled_df)

    # Downloads
    out_xlsx = io.BytesIO()
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        schedule_display.to_excel(writer, sheet_name="Schedule", index=True)
        pd.DataFrame(mapping_rows).to_excel(writer, sheet_name="RoleMapping", index=False)
        report_df.to_excel(writer, sheet_name="AssignmentSummary", index=False)
        cap_df.to_excel(writer, sheet_name="Capacities", index=False)
    out_xlsx.seek(0)

    out_csv = io.StringIO()
    schedule_display.to_csv(out_csv)
    out_csv.seek(0)

    st.download_button("Download Excel (.xlsx)", data=out_xlsx, file_name=f"uKids_schedule_{year}_{month:02d}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download CSV (.csv)", data=out_csv.getvalue(), file_name=f"uKids_schedule_{year}_{month:02d}.csv", mime="text/csv")

else:
    st.info("Upload the two Excel files for a single month and click **Generate Schedule**. ")










