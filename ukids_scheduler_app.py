# Streamlit app for uKids Children Scheduler (Max-Fill, Deterministic) + LowAvailability export + Inline Flags
# Save as ukids_scheduler_app.py and run with: streamlit run ukids_scheduler_app.py

import io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date
from typing import Dict, List, Tuple
import base64
import re
from collections import defaultdict

st.set_page_config(page_title='uKids Scheduler', layout='wide')
st.title('uKids Scheduler')

# ---- Black theme via CSS ----
st.markdown(
    """
    <style>
        body { background-color: #000000; color: white; }
        .stApp { background-color: #000000; }
        .stButton>button, .stDownloadButton>button { background-color: #444; color: white; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Centered logo (optional) ----
try:
    with open("image(1).png", "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
            <div style='text-align: center;'>
                <img src='data:image/png;base64,{encoded}' width='600'>
            </div>
        """, unsafe_allow_html=True)
except FileNotFoundError:
    pass  # logo optional

# ------------------------------
# Config
# ------------------------------
MOST_PEOPLE_RATIO = 0.80   # if >=80% of people reached 2, allow a 3rd assignment to fill gaps

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

# Default capacities (hidden; not shown in UI)
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
    # Add leader rows here if desired, e.g. {"Role": "Age 1 leader", "Capacity": 1},
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

# ------------------------------
# Max-Flow (Dinic) for max-fill scheduling
# ------------------------------
class Dinic:
    def __init__(self, N):
        self.N = N
        self.adj = [[] for _ in range(N)]
               self.level = [0]*self.N
        self.it = [0]*self.N

    def add_edge(self, u, v, cap):
        self.adj[u].append([v, cap, len(self.adj[v])])
        self.adj[v].append([u, 0, len(self.adj[u])-1])

    def bfs(self, s, t):
        from collections import deque
        self.level = [-1]*self.N
        q = deque([s])
        self.level[s] = 0
        while q:
            u = q.popleft()
            for v, cap, rev in self.adj[u]:
                if cap > 0 and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    q.append(v)
        return self.level[t] >= 0

    def dfs(self, u, t, f):
        if u == t:
            return f
        for i in range(self.it[u], len(self.adj[u])):
            self.it[u] = i
            v, cap, rev = self.adj[u][i]
            if cap > 0 and self.level[u] < self.level[v]:
                d = self.dfs(v, t, min(f, cap))
                if d > 0:
                    self.adj[u][i][1] -= d
                    self.adj[v][rev][1] += d
                    return d
        return 0

    def max_flow(self, s, t):
        flow = 0
        INF = 10**9
        while self.bfs(s, t):
            self.it = [0]*self.N
            f = self.dfs(s, t, INF)
            while f > 0:
                flow += f
                f = self.dfs(s, t, INF)
        return flow

def schedule_maxfill(long_df: pd.DataFrame,
                     availability: Dict[str, Dict[pd.Timestamp, bool]],
                     service_dates: List[pd.Timestamp],
                     capacity_df: pd.DataFrame,
                     role_map: Dict[str, List[str]],
                     per_person_cap: int,
                     restrict_people: set = None):
    people = sorted(long_df['person'].unique())
    if restrict_people is not None:
        people = [p for p in people if p in restrict_people]
    people_idx = {p:i for i,p in enumerate(people)}

    # Precompute eligibility best priority per person for each capacity role
    elig_by_person = {}
    for p in people:
        sub = long_df[long_df['person'] == p]
        pr_map = {row['role']: int(row['priority']) for _, row in sub.iterrows()}
        elig_by_person[p] = pr_map

    # Build slot list deterministically
    role_names = [str(r['Role']).strip() for _, r in capacity_df.iterrows()]
    slots = []  # (role, date, slot_idx)
    for role in role_names:
        cap = int(capacity_df.loc[capacity_df['Role'] == role, 'Capacity'].iloc[0])
        for d in service_dates:
            for sidx in range(cap):
                slots.append((role, d, sidx))

    # Person-date pairs
    pd_pairs = set()
    cand_per_slot = {}
    for role, d, sidx in slots:
        mapped_sheet_roles = role_map.get(role, [])
        cands = []
        if mapped_sheet_roles:
            for p in people:
                if not availability.get(p, {}).get(d, False):
                    continue
                prs = [elig_by_person[p].get(rn, None) for rn in mapped_sheet_roles if rn in elig_by_person[p]]
                prs = [pr for pr in prs if pr is not None and pr >= 1]
                if not prs:
                    continue
                best_pr = min(prs)  # 1 best
                cands.append((best_pr, p))
                pd_pairs.add((p, d))
        cands.sort(key=lambda x: (x[0], x[1]))
        cand_per_slot[(role, d, sidx)] = [p for _, p in cands]

    # Build graph
    SLOTS = len(slots)
    pd_index = {pd:i for i,pd in enumerate(sorted(pd_pairs, key=lambda x: (x[1], x[0])))}  # sort by date then person
    src = 0
    slot_base = 1
    pd_base = slot_base + SLOTS
    person_base = pd_base + len(pd_index)
    sink = person_base + len(people)
    N = sink + 1
    din = Dinic(N)

    # source -> slot
    for i, sd in enumerate(slots):
        din.add_edge(src, slot_base + i, 1)

    # slot -> person-date
    for i, sd in enumerate(slots):
        for p in cand_per_slot[sd]:
            pd_id = pd_index[(p, sd[1])]
            din.add_edge(slot_base + i, pd_base + pd_id, 1)

    # person-date -> person
    for (p, d), idx in pd_index.items():
        din.add_edge(pd_base + idx, person_base + people_idx[p], 1)

    # person -> sink
    for p, pi in people_idx.items():
        din.add_edge(person_base + pi, sink, per_person_cap)

    din.max_flow(src, sink)

    # Decode assignments
    assigned = defaultdict(list)  # key: (role, d) -> list of (sidx, name)
    for i, sd in enumerate(slots):
        for v, cap, rev in din.adj[slot_base + i]:
            if v >= pd_base and v < person_base and cap == 0:
                pd_id = v - pd_base
                (pname, d) = list(pd_index.keys())[pd_id]
                assigned[(sd[0], d)].append((sd[2], pname))

    # Assemble schedule_cells with slot order
    schedule_cells = {}
    for role, d, sidx in slots:
        key = (role, d)
        if key not in schedule_cells:
            cap = int(capacity_df.loc[capacity_df['Role'] == role, 'Capacity'].iloc[0])
            schedule_cells[key] = ["" for _ in range(cap)]
    for key, picks in assigned.items():
        role, d = key
        cap = int(capacity_df.loc[capacity_df['Role'] == role, 'Capacity'].iloc[0])
        for sidx, pname in picks:
            if 0 <= sidx < cap:
                schedule_cells[key][sidx] = pname

    # Build counts
    assign_count = defaultdict(int)
    for (role, d), names in schedule_cells.items():
        for n in names:
            if n:
                assign_count[n] += 1

    # Unfilled
    unfilled = []
    for (role, d), names in schedule_cells.items():
        missing = sum(1 for n in names if not n)
        if missing > 0:
            unfilled.append({"Role": role, "Date": d.strftime("%Y-%m-%d"), "MissingCount": missing})
    unfilled_df = pd.DataFrame(unfilled)

    return schedule_cells, dict(assign_count), unfilled_df

def schedule_with_two_then_three(long_df, availability, service_dates, capacity_df, role_map):
    sc1, counts1, unfilled1 = schedule_maxfill(long_df, availability, service_dates, capacity_df, role_map, per_person_cap=2)
    people = list({row['person'] for _, row in long_df.iterrows()})
    ratio_with_two = (sum(1 for p in people if counts1.get(p, 0) >= 2) / max(1, len(people)))

    schedule_cells = sc1
    counts = counts1
    unfilled = unfilled1.copy()

    if not unfilled1.empty and ratio_with_two >= MOST_PEOPLE_RATIO:
        topup_people = {p for p in people if counts1.get(p, 0) == 2}
        sc2, counts2_add, unfilled2 = schedule_maxfill(long_df, availability, service_dates, capacity_df, role_map, per_person_cap=1, restrict_people=topup_people)

        for key, arr in schedule_cells.items():
            add_arr = sc2.get(key, [])
            for i in range(len(arr)):
                if arr[i] == "" and i < len(add_arr) and add_arr[i] != "":
                    arr[i] = add_arr[i]
                    counts[arr[i]] = counts.get(arr[i], 0) + 1

        unfilled = []
        for (role, d), names in schedule_cells.items():
            missing = sum(1 for n in names if not n)
            if missing > 0:
                unfilled.append({"Role": role, "Date": d.strftime("%Y-%m-%d"), "MissingCount": missing})
        unfilled = pd.DataFrame(unfilled)
    else:
        unfilled = unfilled1

    return schedule_cells, counts, unfilled, ratio_with_two

def expand_to_slot_rows(capacity_df: pd.DataFrame, service_dates: List[pd.Timestamp], schedule_cells: Dict[tuple, list]) -> pd.DataFrame:
    rows = []
    index_labels = []
    date_cols = [d.strftime('%Y-%m-%d') for d in service_dates]

    for _, crow in capacity_df.iterrows():
        cap_role = str(crow['Role']).strip()
        cap = int(crow['Capacity'])
        is_volunteers = cap_role.lower().endswith('volunteers')
        base_label = cap_role[:-10].strip() if is_volunteers else cap_role  # remove 'volunteers'

        for slot_idx in range(cap):
            row = {}
            for d in service_dates:
                names = schedule_cells.get((cap_role, d), [])
                row[d.strftime('%Y-%m-%d')] = names[slot_idx] if slot_idx < len(names) else ''
            rows.append(row)
            index_labels.append(base_label)

    disp = pd.DataFrame(rows, columns=date_cols)
    disp.index = index_labels
    return disp

def build_availability_counts_df(long_df: pd.DataFrame, availability: Dict[str, Dict[pd.Timestamp, bool]], service_dates: List[pd.Timestamp]) -> pd.DataFrame:
    """Availability counts for ALL eligible people (from serving positions with priority >=1 on at least one role)."""
    people_pool = sorted(long_df['person'].unique())
    records = []
    for p in people_pool:
        yes_dates = [d for d in service_dates if availability.get(p, {}).get(d, False)]
        records.append({
            "Person": p,
            "YesCount": len(yes_dates),
            "YesDates": ", ".join(sorted(d.strftime('%Y-%m-%d') for d in yes_dates)),
            "MissingToTwo": max(0, 2 - len(yes_dates)),
        })
    return pd.DataFrame(records)

def build_low_availability_df(long_df: pd.DataFrame, availability: Dict[str, Dict[pd.Timestamp, bool]], service_dates: List[pd.Timestamp]) -> pd.DataFrame:
    df = build_availability_counts_df(long_df, availability, service_dates)
    return df[df["YesCount"] < 2].sort_values(["YesCount", "Person"]).reset_index(drop=True)

# ------------------------------
# UI (everything on page)
# ------------------------------
st.subheader("1) Upload files (single month only)")
col1, col2 = st.columns(2)
with col1:
    people_file = st.file_uploader("Serving positions (Excel)", type=["xlsx", "xls"], key="people_file")
with col2:
    responses_file = st.file_uploader("Form responses (Excel)", type=["xlsx", "xls"], key="responses_file")

run_btn = st.button("Generate Schedule", type="primary")

if run_btn:
    if not people_file or not responses_file:
        st.error("Please upload both the serving positions and the form responses files.")
        st.stop()

    people_df = pd.read_excel(people_file)
    responses_df = pd.read_excel(responses_file)

    cap_df = pd.DataFrame(DEFAULT_CAPACITIES)

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

    # Availability diagnostics
    all_avail_df = build_availability_counts_df(long_df, availability, service_dates)
    low_avail_df = all_avail_df[all_avail_df["YesCount"] < 2].copy().reset_index(drop=True)

    role_map = map_capacity_roles_to_sheet(cap_df, role_cols)

    schedule_cells, assign_count, unfilled_df, ratio_with_two = schedule_with_two_then_three(
        long_df, availability, service_dates, cap_df, role_map
    )

    st.success(f"Schedule generated for {date(service_dates[0].year, service_dates[0].month, 1):%B %Y}!")

    st.subheader("2) Schedule (each slot as its own row; dates as columns)")
    slot_disp = expand_to_slot_rows(cap_df, service_dates, schedule_cells)
    st.dataframe(slot_disp, use_container_width=True)

    st.subheader("Assignment Summary")
    # Merge counts with availability info and add Flag
    summary_df = pd.Series(assign_count, name="AssignedCount").rename_axis("Person").reset_index()
    summary_df = summary_df.merge(all_avail_df, on="Person", how="left")
    summary_df["YesCount"] = summary_df["YesCount"].fillna(0).astype(int)
    summary_df["MissingToTwo"] = summary_df["MissingToTwo"].fillna(2).astype(int)
    summary_df["YesDates"] = summary_df["YesDates"].fillna("")
    summary_df["Flag"] = np.where(summary_df["YesCount"] < 2, "⚠️ Low availability", "")
    summary_df = summary_df.sort_values(by=["Flag", "MissingToTwo", "Person"], ascending=[False, False, True])
    st.dataframe(summary_df, use_container_width=True)

    st.info(f"{ratio_with_two:.0%} of people reached 2 assignments. "
            f"{'A 3rd-pass fill was applied.' if ratio_with_two >= MOST_PEOPLE_RATIO else 'No 3rd-pass fill applied (threshold 80%).'}")

    if not unfilled_df.empty:
        st.warning("Some slots could not be filled under the rules:")
        st.dataframe(unfilled_df, use_container_width=True)

    if not low_avail_df.empty:
        st.warning(f"{len(low_avail_df)} people said 'Yes' to fewer than 2 dates (see 'LowAvailability' sheet in the export).")
        st.dataframe(low_avail_df, use_container_width=True)

    # Downloads (export per-slot rows + LowAvailability + enriched AssignmentSummary)
    out_xlsx = io.BytesIO()
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        slot_disp.to_excel(writer, sheet_name="Schedule", index=True)
        summary_df.to_excel(writer, sheet_name="AssignmentSummary", index=False)
        cap_df.to_excel(writer, sheet_name="Capacities", index=False)
        low_avail_df.to_excel(writer, sheet_name="LowAvailability", index=False)
    out_xlsx.seek(0)

    out_csv = io.StringIO()
    slot_disp.to_csv(out_csv)
    out_csv.seek(0)

    st.download_button("Download Excel (.xlsx)", data=out_xlsx, file_name=f"uKids_schedule_{year}_{month:02d}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button("Download CSV (.csv)", data=out_csv.getvalue(), file_name=f"uKids_schedule_{year}_{month:02d}.csv", mime="text/csv")

else:
    st.info("Upload the two Excel files for a single month and click **Generate Schedule**. "
            "The app detects the month automatically from headers like 'Are you available 7 September?'.")
