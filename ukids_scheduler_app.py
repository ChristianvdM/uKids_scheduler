# Streamlit app for uKids Children Scheduler
# Save as ukids_scheduler_app.py and run with: streamlit run ukids_scheduler_app.py

import io
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, date
from typing import Dict, List, Tuple

st.set_page_config(page_title='uKids Scheduler', layout='wide')
st.title('uKids Children Scheduler')

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

# ---------- Helpers ----------
def detect_name_column(df: pd.DataFrame) -> str:
    # Prefer common name headings; fallback to first column
    for pref in ['Name', 'Full name', 'Full names', 'What is your name AND surname?']:
        for c in df.columns:
            if isinstance(c, str) and c.strip().lower() == pref.strip().lower():
                return c
    candidates = [c for c in df.columns if isinstance(c, str) and 'name' in c.lower()]
    return candidates[0] if candidates else df.columns[0]

def is_priority_col(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors='coerce').dropna()
    if len(vals) == 0:
        return False
    return (vals.min() >= 0) and (vals.max() <= 5)

def build_long_df(people_df: pd.DataFrame, name_col: str, role_cols: List[str]) -> pd.DataFrame:
    records = []
    for _, r in people_df.iterrows():
        person = str(r[name_col]).strip()
        if not person or person.lower() == 'nan':
            continue
        for role in role_cols:
            pr = pd.to_numeric(r[role], errors='coerce')
            if pd.isna(pr):
                continue
            pr = int(round(pr))
            if pr >= 1:  # allow only roles with priority 1..5
                records.append({'person': person, 'role': role, 'priority': pr})
    return pd.DataFrame(records)

def parse_availability(responses_df: pd.DataFrame, name_col_resp: str, year: int, month: int):
    # Expect columns like "Are you available 7 September?"
    avail_cols = [c for c in responses_df.columns if isinstance(c, str) and c.strip().lower().startswith('are you available')]
    date_map = {}
    for c in avail_cols:
        digits = ''.join(ch for ch in c if ch.isdigit())
        if not digits:
            continue
        day = int(digits)
        try:
            dt = pd.Timestamp(datetime(year, month, day)).normalize()
        except Exception:
            continue
        date_map[c] = dt
    availability: Dict[str, Dict[pd.Timestamp, bool]] = {}
    for _, row in responses_df.iterrows():
        nm = str(row.get(name_col_resp, '')).strip()
        if not nm or nm.lower() == 'nan':
            continue
        availability.setdefault(nm, {})
        for col, dt in date_map.items():
            ans = str(row.get(col, '')).strip().lower()
            is_yes = ans in {'yes', 'y', 'true', 'available'}
            availability[nm][dt] = is_yes
    dates_sorted = sorted(set(date_map.values()))
    return availability, dates_sorted

def schedule_greedy(long_df: pd.DataFrame, availability: Dict[str, Dict[pd.Timestamp, bool]],
                    service_dates, roles, mode: str, seed: int = 123):
    rng = np.random.default_rng(seed)
    target = 2
    assign_count = {p: 0 for p in long_df['person'].unique()}
    schedule = pd.DataFrame(index=service_dates, columns=roles)

    # Precompute per-slot eligibility
    slots = [(d, role) for d in service_dates for role in roles]
    slot_eligible = {}
    for (d, role) in slots:
        sub = long_df[long_df['role'] == role]
        elig = []
        for _, rr in sub.iterrows():
            p = rr['person']
            pr = rr['priority']
            if availability.get(p, {}).get(d, False):
                elig.append((p, pr))
        slot_eligible[(d, role)] = elig

    if mode == 'max2':
        cap = 2
        for d in service_dates:
            assigned_today = set()
            for role in roles:
                elig = [(p, pr) for (p, pr) in slot_eligible[(d, role)]
                        if assign_count[p] < cap and p not in assigned_today]
                if not elig:
                    continue
                weights = np.array([pr for _, pr in elig], dtype=float) ** 2.0
                weights = weights / weights.sum()
                chosen = rng.choice([p for p, _ in elig], p=weights)
                schedule.loc[d, role] = chosen
                assign_count[chosen] += 1
                assigned_today.add(chosen)
        return schedule, assign_count

    # "exactly 2" mode — two-pass balancing
    for d in service_dates:
        assigned_today = set()
        for role in roles:
            elig = [(p, pr) for (p, pr) in slot_eligible[(d, role)]
                    if assign_count[p] < target and p not in assigned_today]
            if not elig:
                continue
            # pick among those with lowest current count (0 first), then higher priority
            min_count = min(assign_count[p] for p, _ in elig)
            pool = [(p, pr) for (p, pr) in elig if assign_count[p] == min_count]
            pool.sort(key=lambda x: (-x[1], rng.random()))
            chosen = pool[0][0]
            schedule.loc[d, role] = chosen
            assign_count[chosen] += 1
            assigned_today.add(chosen)

    # Fill remaining empties with people still under target (avoid same-day double-book)
    for d in service_dates:
        assigned_today = set([v for v in schedule.loc[d, :].dropna().values])
        for role in roles:
            if pd.notna(schedule.loc[d, role]):
                continue
            elig = [(p, pr) for (p, pr) in slot_eligible[(d, role)]
                    if assign_count[p] < target and p not in assigned_today]
            if not elig:
                continue
            elig.sort(key=lambda x: (assign_count[x[0]], -x[1], rng.random()))
            for p, pr in elig:
                if assign_count[p] < target:
                    schedule.loc[d, role] = p
                    assign_count[p] += 1
                    assigned_today.add(p)
                    break

    return schedule, assign_count

def make_report(assign_count, availability, long_df, service_dates):
    series = pd.Series(assign_count, name='AssignedCount').sort_values(ascending=False)
    summary_df = series.reset_index().rename(columns={'index': 'Person'})
    potential = {p: 0 for p in assign_count.keys()}
    for p in potential.keys():
        potential[p] = sum(1 for d in service_dates if availability.get(p, {}).get(d, False))
    pot_df = pd.DataFrame({'Person': list(potential.keys()), 'AvailableYesDates': list(potential.values())})
    return summary_df.merge(pot_df, on='Person', how='left')

# ---------- Sidebar ----------
with st.sidebar:
    st.header('Inputs')
    people_file = st.file_uploader('Serving positions (Excel)', type=['xlsx', 'xls'])
    responses_file = st.file_uploader('Form responses (Excel)', type=['xlsx', 'xls'])
    year = st.number_input('Year', min_value=2024, max_value=2030, value=date.today().year, step=1)
    month = st.number_input('Month (1-12)', min_value=1, max_value=12, value=date.today().month, step=1)
    mode = st.selectbox('Assignment rule', ['Exactly 2 per person', 'Max 2 per person'], index=0)
    seed = st.number_input('Random seed', value=123, step=1)
    run_btn = st.button('Generate Schedule')

# ---------- Main ----------
if run_btn:
    if not people_file or not responses_file:
        st.error('Please upload both the serving positions and the form responses files.')
        st.stop()

    people_df = pd.read_excel(people_file)
    responses_df = pd.read_excel(responses_file)

    name_col_people = detect_name_column(people_df)
    name_col_resp = detect_name_column(responses_df)

    role_cols = [c for c in people_df.columns if c != name_col_people and is_priority_col(people_df[c])]
    if not role_cols:
        st.error('No role columns with priorities (0-5) detected.')
        st.stop()

    long_df = build_long_df(people_df, name_col_people, role_cols)
    if long_df.empty:
        st.error('No eligible assignments found (all priorities are 0 or missing).')
        st.stop()

    availability, service_dates = parse_availability(responses_df, name_col_resp, int(year), int(month))
    if not service_dates:
        st.error('Could not detect any service dates from the responses file.')
        st.stop()

    schedule, assign_count = schedule_greedy(
        long_df, availability, service_dates, role_cols,
        mode='exact2' if mode.startswith('Exactly') else 'max2', seed=int(seed)
    )

    st.success('Schedule generated!')
    st.write('### Schedule')
    st.dataframe(schedule.reset_index().rename(columns={'index': 'Date'}))

    report_df = make_report(assign_count, availability, long_df, service_dates)
    st.write('### Assignment Summary')
    st.dataframe(report_df)

    out_xlsx = io.BytesIO()
    with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
        schedule.reset_index().rename(columns={'index': 'Date'}).to_excel(writer, sheet_name='Schedule', index=False)
        report_df.to_excel(writer, sheet_name='AssignmentSummary', index=False)
        pd.DataFrame({'Roles': role_cols}).to_excel(writer, sheet_name='Roles', index=False)
    out_xlsx.seek(0)

    out_csv = io.StringIO()
    schedule.reset_index().rename(columns={'index': 'Date'}).to_csv(out_csv, index=False)
    out_csv.seek(0)

    st.download_button('Download Excel (.xlsx)', data=out_xlsx, file_name=f'uKids_schedule_{year}_{month:02d}.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    st.download_button('Download CSV (.csv)', data=out_csv.getvalue(), file_name=f'uKids_schedule_{year}_{month:02d}.csv', mime='text/csv')

    under = [p for p, c in assign_count.items() if c < 2]
    over = [p for p, c in assign_count.items() if c > 2]
    exact = [p for p, c in assign_count.items() if c == 2]
    st.write('### Diagnostics')
    st.write(f'People with < 2 assignments: {len(under)}')
    st.write(f'People with = 2 assignments: {len(exact)}')
    st.write(f'People with > 2 assignments: {len(over)} (should be 0)')
else:
    st.info('Upload the two Excel files on the left, set month/year and rule, then click **Generate Schedule**.')


