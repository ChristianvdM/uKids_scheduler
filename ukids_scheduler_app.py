
# Streamlit app for uKids Children Scheduler (Max-Fill, Best-of-N Search)
# - Priorities are ELIGIBILITY only (0=ineligible). We do NOT prefer 1 over 5; we just try to fill.
# - Best-of-N randomized tie-breaking to find arrangement with the FEWEST unfilled roles.
# - Still enforces: max 2 per person (then optional 3rd pass if >=80% hit 2).
# - Styled export matching the "August 2025" template (headers, merged sections, widths, wrapping).

import io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date
from typing import Dict, List, Tuple, Set
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
        .stDataFrame { font-size: 14px; }
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
MOST_PEOPLE_RATIO = 0.80      # if >=80% reached 2, allow a 3rd assignment
BEST_OF_N = 25# randomized tie-break restarts; higher = better chance to minimize unfilled
RNG_SEED_BASE = 20250811      # base seed so results are stable per run, but varied across tries

SEARCH_RUNS = 25

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

# Capacities (leaders included). Internal only; not shown/exported.
DEFAULT_CAPACITIES = [
    {"Role": "Oversight", "Capacity": 1},
    {"Role": "Main Director", "Capacity": 1},
    {"Role": "Director Roaming Inside", "Capacity": 1},
    {"Role": "Director Roaming Outside", "Capacity": 1},

    {"Role": "Age 1 leader", "Capacity": 1},
    {"Role": "Age 1 volunteers", "Capacity": 5},
    {"Role": "Age 1 nappies", "Capacity": 1},
    {"Role": "Age 1 bags girls", "Capacity": 1},
    {"Role": "Age 1 bags boys", "Capacity": 1},

    {"Role": "Age 2 leader", "Capacity": 1},
    {"Role": "Age 2 volunteers", "Capacity": 4},
    {"Role": "Age 2 nappies", "Capacity": 1},
    {"Role": "Age 2 bags girls", "Capacity": 1},
    {"Role": "Age 2 bags boys", "Capacity": 1},

    {"Role": "Age 3 leader", "Capacity": 1},
    {"Role": "Age 3 volunteers", "Capacity": 4},
    {"Role": "Age 3 bags", "Capacity": 1},

    {"Role": "Age 4 leader", "Capacity": 1},
    {"Role": "Age 4 volunteers", "Capacity": 4},

    {"Role": "Age 5 leader", "Capacity": 1},
    {"Role": "Age 5 volunteers", "Capacity": 3},

    {"Role": "Age 6 leader", "Capacity": 1},
    {"Role": "Age 6 volunteers", "Capacity": 3},

    {"Role": "Age 7 leader", "Capacity": 1},
    {"Role": "Age 7 volunteers", "Capacity": 2},

    {"Role": "Age 8 leader", "Capacity": 1},
    {"Role": "Age 8 volunteers", "Capacity": 2},

    {"Role": "Age 9 leader", "Capacity": 1},
    {"Role": "Age 9 volunteers", "Capacity": 2},

    {"Role": "Age 10 leader", "Capacity": 1},
    {"Role": "Age 10 volunteers", "Capacity": 1},

    {"Role": "Age 11 leader", "Capacity": 1},
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
    def tokenize(s: str):
        return normalize(s).split()

    def matches_tokens(cap_tokens, sheet_tokens):
        IGNORE = {"volunteers", "volunteer", "helpers", "helper"}
        sheet_set = set(sheet_tokens)
        for tok in cap_tokens:
            if tok in IGNORE:
                continue
            base = tok[:-1] if tok.endswith('s') else tok
            variants = {tok, base, base + 's'}
            if base.endswith('ie'):
                variants.add(base[:-2] + 'y')
            if base.endswith('y'):
                variants.add(base[:-1] + 'ies')
            if not any(v in sheet_set for v in variants):
                return False
        return True

    sheet_norm_map = {normalize(c): c for c in people_role_cols}
    sheet_token_map = {k: tokenize(k) for k in sheet_norm_map.keys()}

    mapping: Dict[str, List[str]] = {}
    for _, row in cap_df.iterrows():
        cap_role = str(row["Role"]).strip()
        cap_norm = normalize(cap_role)
        cap_tokens = tokenize(cap_norm)
        matches = []
        if cap_norm in sheet_norm_map:
            matches = [sheet_norm_map[cap_norm]]
        else:
            for k_norm, original in sheet_norm_map.items():
                if matches_tokens(cap_tokens, sheet_token_map[k_norm]):
                    matches.append(original)
        mapping[cap_role] = sorted(set(matches))
    return mapping

# ------------------------------
# Max-Flow (Dinic) with randomized tie-breaking
# ------------------------------
class Dinic:
    def __init__(self, N: int):
        self.N = N
        self.adj = [[] for _ in range(N)]
        self.level = [0] * N
        self.it = [0] * N

    def add_edge(self, u: int, v: int, cap: int):
        self.adj[u].append([v, cap, len(self.adj[v])])
        self.adj[v].append([u, 0, len(self.adj[u]) - 1])

    def bfs(self, s: int, t: int) -> bool:
        from collections import deque
        self.level = [-1] * self.N
        q = deque([s])
        self.level[s] = 0
        while q:
            u = q.popleft()
            for v, cap, rev in self.adj[u]:
                if cap > 0 and self.level[v] < 0:
                    self.level[v] = self.level[u] + 1
                    q.append(v)
        return self.level[t] >= 0

    def dfs(self, u: int, t: int, f: int) -> int:
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

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        INF = 10 ** 9
        while self.bfs(s, t):
            self.it = [0] * self.N
            f = self.dfs(s, t, INF)
            while f > 0:
                flow += f
                f = self.dfs(s, t, INF)
        return flow

def run_flow_once(long_df, availability, service_dates, capacity_df, role_map, per_person_cap, rng=None):
    # Build people list and indices
    people = sorted(long_df['person'].unique())
    people_idx = {p: i for i, p in enumerate(people)}

    # Eligibility map (priority only used to compute optional score; not to filter except pr>=1)
    elig_by_person = {}
    for p in people:
        sub = long_df[long_df['person'] == p]
        pr_map = {row['role']: int(row['priority']) for _, row in sub.iterrows()}
        elig_by_person[p] = pr_map

    # Build slots (role, date, slot_idx) and optionally shuffle for tie-breaking variety
    role_names = [str(r['Role']).strip() for _, r in capacity_df.iterrows()]
    slots = []
    for role in role_names:
        cap = int(capacity_df.loc[capacity_df['Role'] == role, 'Capacity'].iloc[0])
        for d in service_dates:
            for sidx in range(cap):
                slots.append((role, d, sidx))
    if rng is not None:
        rng.shuffle(slots)

    # Candidate edges: any pr>=1 => eligible; order optionally shuffled
    pd_pairs = set()
    cand_per_slot = {}
    for role, d, sidx in slots:
        mapped = role_map.get(role, [])
        cands = []
        if mapped:
            for p in people:
                if not availability.get(p, {}).get(d, False):
                    continue
                # at least one eligible mapped role with pr>=1
                ok = any((elig_by_person[p].get(rn, 0) >= 1) for rn in mapped if rn in elig_by_person[p])
                if ok:
                    # record min priority just for scoring later; not used to decide edges
                    best_pr = min((elig_by_person[p].get(rn, 99) for rn in mapped))
                    cands.append((best_pr, p))
                    pd_pairs.add((p, d))
        # shuffle to alter Dinic's tie-breaks
        if rng is not None:
            rng.shuffle(cands)
        cand_per_slot[(role, d, sidx)] = [p for _, p in cands]

    # Build graph
    SLOTS = len(slots)
    pd_list = sorted(pd_pairs, key=lambda x: (x[1], x[0]))
    pd_index = {pd: i for i, pd in enumerate(pd_list)}

    src = 0
    slot_base = 1
    pd_base = slot_base + SLOTS
    person_base = pd_base + len(pd_index)
    sink = person_base + len(people)
    N = sink + 1
    din = Dinic(N)

    # source -> slot
    for i in range(SLOTS):
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

    flow = din.max_flow(src, sink)

    # Decode assignments
    assigned = defaultdict(list)
    for i, sd in enumerate(slots):
        for v, cap, rev in din.adj[slot_base + i]:
            if pd_base <= v < person_base and cap == 0:
                pd_id = v - pd_base
                pname, d = pd_list[pd_id]
                assigned[(sd[0], d)].append((sd[2], pname))

    # Assemble schedule_cells
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

    # Metrics
    unfilled = 0
    priority_sum = 0
    for (role, d), names in schedule_cells.items():
        cap = int(capacity_df.loc[capacity_df['Role'] == role, 'Capacity'].iloc[0])
        unfilled += max(0, cap - sum(1 for n in names if n))
        # sum minimal priority each assigned got for that role/date (for tie-break scoring only)
        mapped = role_map.get(role, [])
        for n in names:
            if not n: 
                continue
            best_pr = min((elig_by_person[n].get(rn, 99) for rn in mapped))
            priority_sum += best_pr

    # Counts
    assign_count = defaultdict(int)
    for (role, d), names in schedule_cells.items():
        for n in names:
            if n:
                assign_count[n] += 1

    return schedule_cells, dict(assign_count), unfilled, priority_sum

def best_of_n(long_df, availability, service_dates, capacity_df, role_map, per_person_cap, n_tries=BEST_OF_N, seed=RNG_SEED_BASE):
    best = None
    rng = np.random.default_rng(seed)
    for t in range(n_tries):
        local_rng = np.random.default_rng(int(seed + t*9973))
        sc, counts, unfilled, prsum = run_flow_once(long_df, availability, service_dates, capacity_df, role_map, per_person_cap, rng=local_rng)
        if (best is None) or (unfilled < best[2]) or (unfilled == best[2] and prsum < best[3]):
            best = (sc, counts, unfilled, prsum)
    return best

def schedule_with_two_then_three(long_df, availability, service_dates, capacity_df, role_map):
    sc1, counts1, unfilled1, prsum1 = best_of_n(long_df, availability, service_dates, capacity_df, role_map, per_person_cap=2)
    people = list({row['person'] for _, row in long_df.iterrows()})
    ratio_with_two = (sum(1 for p in people if counts1.get(p, 0) >= 2) / max(1, len(people)))

    schedule_cells = sc1
    counts = counts1

    if unfilled1 > 0 and ratio_with_two >= MOST_PEOPLE_RATIO:
        topup_people = {p for p in people if counts.get(p, 0) == 2}
        # Restrict long_df to topup_people for the 3rd pass
        mask = long_df['person'].isin(list(topup_people))
        long_df_topup = long_df[mask].copy()
        sc2, counts2, unfilled2, prsum2 = best_of_n(long_df_topup, availability, service_dates, capacity_df, role_map, per_person_cap=1)
        # merge sc2 into schedule_cells wherever sc1 has blanks
        for key, arr in schedule_cells.items():
            add_arr = sc2.get(key, [])
            for i in range(len(arr)):
                if arr[i] == "" and i < len(add_arr) and add_arr[i] != "":
                    arr[i] = add_arr[i]
                    counts[arr[i]] = counts.get(arr[i], 0) + 1

    # After optional top-up, compute leftover unfilled
    unfilled_rows = []
    for (role, d), names in schedule_cells.items():
        cap = int(capacity_df.loc[capacity_df['Role'] == role, 'Capacity'].iloc[0])
        missing = max(0, cap - sum(1 for n in names if n))
        if missing:
            unfilled_rows.append({"Role": role, "Date": d.strftime("%Y-%m-%d"), "MissingCount": missing})
    unfilled_df = pd.DataFrame(unfilled_rows)

    return schedule_cells, counts, unfilled_df, ratio_with_two

def group_label_for_age(age: int) -> str:
    if age <= 3:
        return "Babies"
    if age <= 6:
        return "Pre-School"
    if age <= 8:
        return "Elementary"
    return "uGroup"

def export_schedule_like_template(writer, schedule_cells, capacity_df, service_dates):
    wb = writer.book
    ws = wb.add_worksheet("Schedule")
    writer.sheets["Schedule"] = ws

    # Colors from your August 2025 sheet
    role_header_fill = '#BDC0BF'  # Column A header
    date_header_fill = '#E3167D'  # Date headers

    # Formats
    header_role_fmt = wb.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'bottom': 1, 'bg_color': role_header_fill, 'border': 1})
    header_date_fmt = wb.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'bottom': 1, 'bg_color': date_header_fill, 'border': 1, 'font_color': 'white'})
    role_fmt = wb.add_format({'text_wrap': True, 'valign': 'top', 'border': 1})
    name_fmt = wb.add_format({'text_wrap': True, 'valign': 'top', 'border': 1})
    section_fmt = wb.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'bg_color': '#D9D9D9', 'border': 1})
    spacer_fmt = wb.add_format({'bg_color': '#FFFFFF'})

    # Column widths approximated from template (A wide, date cols medium)
    ws.set_column(0, 0, 21.57, role_fmt)   # Role col
    ws.set_column(1, 1 + len(service_dates), 24.43, name_fmt)  # Date cols default

    # Header row
    ws.write(0, 0, "Role", header_role_fmt)
    for j, d in enumerate(service_dates, start=1):
        ws.write_datetime(0, j, d.to_pydatetime(), header_date_fmt)

    def add_admin_rows(rows):
        admin_roles = ["Oversight", "Main Director", "Director Roaming Inside", "Director Roaming Outside"]
        for r in admin_roles:
            rows.append(("role", r))
        rows.append(("spacer", ""))

    def add_age_block(rows, age: int):
        group = group_label_for_age(age)
        if age in (1, 4, 7, 9):
            rows.append(("section", group))

        if age <= 3:
            leader_label = f"Babies Leader Age {age}"
        elif age <= 6:
            leader_label = f"Pre-School Leader Age {age}"
        elif age <= 8:
            leader_label = f"Elementary Leader Age {age}"
        else:
            leader_label = f"uGroup Age {age}"
        rows.append(("role", leader_label))

        vol_cap = int(capacity_df.loc[capacity_df["Role"] == f"Age {age} volunteers", "Capacity"].iloc[0])
        for _ in range(vol_cap):
            rows.append(("role", f"Age {age}"))

        if age <= 2:
            rows.append(("role", f"Age {age} nappies"))
            rows.append(("role", f"Age {age} bags girls"))
            rows.append(("role", f"Age {age} bags boys"))
        elif age == 3:
            rows.append(("role", "Age 3 bags"))

        rows.append(("spacer", ""))

    def add_special_needs(rows):
        rows.append(("section", "Special Needs"))
        cap = int(capacity_df.loc[capacity_df["Role"] == "Special needs volunteers", "Capacity"].iloc[0])
        for _ in range(cap):
            rows.append(("role", "Special Needs"))

    ordered_rows = []
    add_admin_rows(ordered_rows)
    for a in range(1, 12):
        add_age_block(ordered_rows, a)
    add_special_needs(ordered_rows)

    def names_for_label_and_date(label: str, d):
        if (label, d) in schedule_cells:
            return schedule_cells[(label, d)]
        m = re.match(r"^(Babies Leader Age|Pre-School Leader Age|Elementary Leader Age|uGroup Age) (\d+)$", label)
        if m:
            age = int(m.group(2))
            cap_role = f"Age {age} leader"
            return schedule_cells.get((cap_role, d), [])
        m2 = re.match(r"^Age (\d+)$", label)
        if m2:
            age = int(m2.group(1))
            cap_role = f"Age {age} volunteers"
            return schedule_cells.get((cap_role, d), [])
        if label.lower().startswith("age") and "nappies" in label.lower():
            age = int(re.findall(r"\d+", label)[0])
            cap_role = f"Age {age} nappies"
            return schedule_cells.get((cap_role, d), [])
        if label.lower().startswith("age") and "bags girls" in label.lower():
            age = int(re.findall(r"\d+", label)[0])
            cap_role = f"Age {age} bags girls"
            return schedule_cells.get((cap_role, d), [])
        if label.lower().startswith("age") and "bags boys" in label.lower():
            age = int(re.findall(r"\d+", label)[0])
            cap_role = f"Age {age} bags boys"
            return schedule_cells.get((cap_role, d), [])
        if label.lower() == "age 3 bags":
            return schedule_cells.get(("Age 3 bags", d), [])
        if label == "Special Needs":
            return schedule_cells.get(("Special needs volunteers", d), [])
        return []

    excel_row = 1
    for kind, label in ordered_rows:
        if kind == "section":
            ws.merge_range(excel_row, 0, excel_row, len(service_dates), f"  {label}", section_fmt)
            excel_row += 1
            continue
        if kind == "spacer":
            ws.write(excel_row, 0, "", spacer_fmt)
            excel_row += 1
            continue

        ws.write(excel_row, 0, label, role_fmt)
        for j, d in enumerate(service_dates, start=1):
            if re.fullmatch(r"Age \d+", label):
                ws.write_string(excel_row, j, "", name_fmt)  # placeholder
            else:
                names = names_for_label_and_date(label, d)
                ws.write_string(excel_row, j, ", ".join([n for n in names if n]), name_fmt)
        excel_row += 1

    # Second pass for volunteers: put one slot per repeated "Age N" row
    row_map = []
    excel_row2 = 1
    for kind, label in ordered_rows:
        if kind in ("section", "spacer"):
            continue
        row_map.append((excel_row2, label))
        excel_row2 += 1

    from collections import defaultdict as dd
    per_date_written_count = dd(lambda: dd(int))
    for row_idx, label in row_map:
        if re.fullmatch(r"Age \d+", label):
            for j, d in enumerate(service_dates, start=1):
                names = names_for_label_and_date(label, d)
                k = per_date_written_count[j][label]
                val = names[k] if k < len(names) and names[k] else ""
                ws.write_string(row_idx, j, val, name_fmt)
                per_date_written_count[j][label] += 1

    return ws

def expand_to_slot_rows(capacity_df: pd.DataFrame, service_dates: List[pd.Timestamp], schedule_cells: Dict[tuple, list]) -> pd.DataFrame:
    rows = []
    index_labels = []
    date_cols = [d.strftime('%Y-%m-%d') for d in service_dates]

    for _, crow in capacity_df.iterrows():
        cap_role = str(crow['Role']).strip()
        cap = int(crow['Capacity'])
        is_volunteers = cap_role.lower().endswith('volunteers')
        base_label = cap_role[:-10].strip() if is_volunteers else cap_role

        if cap_role.endswith('leader'):
            age_num = ''.join([c for c in cap_role if c.isdigit()])
            if age_num:
                n = int(age_num)
                if n <= 3:
                    base_label = f"Babies Leader Age {n}"
                elif n <= 6:
                    base_label = f"Pre-School Leader Age {n}"
                elif n <= 8:
                    base_label = f"Elementary Leader Age {n}"
                else:
                    base_label = f"uGroup Age {n}"

        if cap_role.lower().startswith('special needs'):
            base_label = "Special Needs"

        for slot_idx in range(cap):
            row = {}
            for d in service_dates:
                names = schedule_cells.get((cap_role, d), [])
                row[d.strftime('%Y-%m-%d')] = names[slot_idx] if slot_idx < len(names) else ''
            rows.append(row)
            index_labels.append(base_label)

    disp = pd.DataFrame(rows, columns=date_cols)
    disp.index = index_labels
    disp.index.name = "Role"
    return disp

def build_availability_counts_df(long_df: pd.DataFrame, availability: Dict[str, Dict[pd.Timestamp, bool]], service_dates: List[pd.Timestamp]) -> pd.DataFrame:
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
    return df[df["YesCount"] <= 2].sort_values(["YesCount", "Person"]).reset_index(drop=True)

# ------------------------------
# UI (everything on page)
# ------------------------------
st.subheader("1) Upload files (single month only)")
col1, col2 = st.columns(2)
with col1:
    people_file = st.file_uploader("Serving positions (Excel)", type=["xlsx", "xls"], key="people_file")
with col2:
    responses_file = st.file_uploader("Form responses (Excel)", type=["xlsx", "xls"], key="responses_file")

# Optional: allow changing Best-of-N for deeper search
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

    all_avail_df = build_availability_counts_df(long_df, availability, service_dates)
    low_avail_df = all_avail_df[all_avail_df["YesCount"] <= 2].copy().reset_index(drop=True)

    role_map = map_capacity_roles_to_sheet(cap_df, role_cols)

    schedule_cells, assign_count, unfilled_df, ratio_with_two = schedule_with_two_then_three(
        long_df, availability, service_dates, cap_df, role_map
    )

    st.success(f"Schedule generated for {date(service_dates[0].year, service_dates[0].month, 1):%B %Y}!")

    st.subheader("2) Schedule (preview)")
    slot_disp = expand_to_slot_rows(cap_df, service_dates, schedule_cells)
    st.dataframe(slot_disp, use_container_width=True)

    st.subheader("Assignment Summary")
    summary_df = pd.Series(assign_count, name="AssignedCount").rename_axis("Person").reset_index()
    st.dataframe(summary_df.sort_values(["AssignedCount","Person"], ascending=[False, True]), use_container_width=True)

    if not unfilled_df.empty:
        st.warning("Some slots could not be filled under the rules:")
        st.dataframe(unfilled_df, use_container_width=True)

    if not low_avail_df.empty:
        st.warning(f"People not available for more than 2 dates (YesCount ≤ 2): {len(low_avail_df)}")
        st.dataframe(low_avail_df, use_container_width=True)

    out_xlsx = io.BytesIO()
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter", datetime_format="dd mmm") as writer:
        export_schedule_like_template(writer, schedule_cells, cap_df, service_dates)

        simple_summary = summary_df[["Person", "AssignedCount"]].copy()
        simple_summary.to_excel(writer, sheet_name="AssignmentSummary", index=False)
        wb = writer.book
        ws2 = writer.sheets["AssignmentSummary"]
        wrap = wb.add_format({"text_wrap": True, "valign": "top"})
        ws2.set_column(0, 0, 28, wrap)
        ws2.set_column(1, 1, 14, wrap)

        if not low_avail_df.empty:
            low_avail_df.to_excel(writer, sheet_name="LowAvailability", index=False)
            ws3 = writer.sheets["LowAvailability"]
            ws3.set_column(0, 0, 28, wrap)
            ws3.set_column(1, 3, 20, wrap)

    out_xlsx.seek(0)
    st.download_button("Download Excel (.xlsx)", data=out_xlsx, file_name=f"uKids_schedule_{year}_{month:02d}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Upload the two Excel files for a single month and click **Generate Schedule**. "
            "The app detects the month automatically from headers like 'Are you available 7 September?'.")
