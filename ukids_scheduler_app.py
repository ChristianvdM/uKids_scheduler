import pandas as pd
from collections import defaultdict

def build_slot_plan():
    """
    Define how many ROWS (independent slots) each position gets.
    Keys should match your column names in the positions sheet.
    Leaders are included here (1 row each).
    """
    slot_plan = {
        # Age 1
        "Age 1 leader": 1,
        "Age 1 classroom": 5,        # volunteers (5)
        "Age 1 nappies": 1,
        "Age 1 bags girls": 1,
        "Age 1 bags boys": 1,

        # Age 2
        "Age 2 leader": 1,
        "Age 2 classroom": 4,        # volunteers (4)
        "Age 2 nappies": 1,
        "Age 2 bags girls": 1,
        "Age 2 bags boys": 1,

        # Age 3
        "Age 3 leader": 1,
        "Age 3 classroom": 4,        # volunteers (4)
        "Age 3 bags": 1,

        # Age 4
        "Age 4 leader": 1,
        "Age 4 classroom": 4,

        # Age 5
        "Age 5 leader": 1,
        "Age 5 classroom": 3,

        # Age 6
        "Age 6 leader": 1,
        "Age 6 classroom": 3,

        # Age 7
        "Age 7 leader": 1,
        "Age 7 classroom": 2,

        # Age 8
        "Age 8 leader": 1,
        "Age 8 classroom": 2,

        # Age 9 (you listed “1 age 9 volunteers” twice; we’ll model as two separate 1‑slot rows)
        "Age 9 leader": 1,
        "Age 9 classroom A": 1,      # volunteers (1)
        "Age 9 classroom B": 1,      # volunteers (1)

        # Age 10
        "Age 10 leader": 1,
        "Age 10 classroom": 1,

        # Age 11
        "Age 11 leader": 1,
        "Age 11 classroom": 1,

        # Special Needs
        "Special needs leader": 1,
        "Special needs classroom": 2,  # volunteers (2)
    }
    return slot_plan

def expand_roles_to_slots(slot_plan):
    """
    Expand role names to concrete slot row labels.
    E.g. "Age 1 classroom": 5 -> ["Age 1 classroom #1", ..., "#5"]
    Single-slot roles keep their base name (no #1 suffix).
    """
    slot_rows = []
    slot_index = {}  # map slot row label -> base role name
    for role, n in slot_plan.items():
        if n <= 0:
            continue
        if n == 1:
            label = role
            slot_rows.append(label)
            slot_index[label] = role
        else:
            for i in range(1, n+1):
                label = f"{role} #{i}"
                slot_rows.append(label)
                slot_index[label] = role
    return slot_rows, slot_index

def build_eligibility(long_df):
    """
    Convert long_df (person, role, priority) to:
    eligibility[person] = set of role names they can do (priority >=1)
    NB: priority==0 means NOT eligible.
    """
    elig = defaultdict(set)
    for _, r in long_df.iterrows():
        person = str(r["person"]).strip()
        role = str(r["role"]).strip()
        pr = int(r["priority"])
        if pr >= 1:        # 0 is “not an option”
            elig[person].add(role)
    return elig

def schedule_by_slots(long_df, availability, service_dates, max_assignments_per_person=2):
    """
    Core scheduler:
    - Every slot row is an independent 1-person position per date
    - Ignores preference ordering (except priority 0 = ineligible)
    - Caps each person at 2 assignments total across the month
    - Tries to fill *everything* greedily date-by-date, slot-by-slot
    - If a slot’s base role is not found in a person’s eligible roles, they’re skipped
    """
    slot_plan = build_slot_plan()
    slot_rows, slot_to_role = expand_roles_to_slots(slot_plan)
    eligibility = build_eligibility(long_df)

    # Count assignments per person (to enforce max 2)
    assign_count = defaultdict(int)

    # Initialize output structure: dict[(slot_row, date)] = assigned_name or ""
    grid = {(slot, d): "" for slot in slot_rows for d in service_dates}

    # Precompute available people list (intersection of availability and eligibility existance)
    people = sorted(set(eligibility.keys()) & set(availability.keys()))

    # For fairness, we’ll always pick the person with the lowest total assigned so far,
    # and avoid double-booking on the same date.
    for d in service_dates:
        assigned_today = set()
        for slot_row in slot_rows:
            base_role = slot_to_role[slot_row]

            # Gather candidates
            cands = []
            for p in people:
                if assign_count[p] >= max_assignments_per_person:
                    continue
                if p in assigned_today:
                    continue
                if not availability.get(p, {}).get(d, False):
                    continue
                # Person must be eligible for the *base role* (match by startswith to handle classroom/bags/lists)
                # We allow some flexible matching: exact, or if any eligible role
                # in sheet contains the base role tokens (helps with “classroom” vs “volunteers” wording).
                elig_roles = eligibility.get(p, set())
                ok = False
                if base_role in elig_roles:
                    ok = True
                else:
                    # try forgiving match (normalize both)
                    def norm(s): return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()
                    nb = norm(base_role)
                    for er in elig_roles:
                        if norm(er) == nb:
                            ok = True
                            break
                if ok:
                    cands.append(p)

            # pick the candidate with the fewest total assignments so far (greedy fill-first)
            if cands:
                cands.sort(key=lambda name: assign_count[name])
                chosen = cands[0]
                grid[(slot_row, d)] = chosen
                assign_count[chosen] += 1
                assigned_today.add(chosen)
            # else it stays "" (unfilled)

    # Build DataFrame: rows = slot rows, columns = dates (YYYY-MM-DD)
    cols = [d.strftime("%Y-%m-%d") for d in service_dates]
    out = pd.DataFrame(index=slot_rows, columns=cols)
    for (slot_row, d), name in grid.items():
        out.loc[slot_row, d.strftime("%Y-%m-%d")] = name

    # Fill NaNs with ""
    out = out.fillna("")
    return out, dict(assign_count)

# ── Example wiring inside your app after you have long_df, availability, service_dates ──
# schedule_df, assign_count = schedule_by_slots(long_df, availability, service_dates)
# st.dataframe(schedule_df, use_container_width=True)
# …then write schedule_df to Excel like you already do.
