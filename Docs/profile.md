# OrbixAI — Profile Intake

> Fill in what you're comfortable with; **leave blanks or delete lines you want to
> skip**. This loads into your **local** Neo4j memory graph (gitignored — never
> committed). Dates as `YYYY-MM-DD`. When done, tell me and I'll write it in via the
> resolve→MERGE protocol (deduped, no duplicates).

```yaml
# ── YOU (the :User singleton) ──────────────────────────────
you:
  name: Kashyap K
  email: kashyapk1305@gmail.com
  other_emails: [pes2ug23cs263@pesu.pes.edu]             # work / college email(s)
  phone: 9686097288
  city:  Banglore                      # where you live now
  home_address: 
  work_address:
  birthday:  13-01-2005               # YYYY-MM-DD
  timezone: Asia/Kolkata
  blood_group: o+
  languages: [English , Kannada, Hindi, Tulu]                # e.g. [English, Kannada, Hindi]

# ── FAMILY (:Person:Family) ────────────────────────────────
# relation: mother | father | brother | sister | spouse | grandparent | ...
family:
  - name: Roopashree K
    relation: mother
    city: Puttur
    birthday: 08-08-1983
    phone: 
  # add more blocks as needed (copy the 5 lines above)

# ── FRIENDS / KEY CONTACTS (:Person) ───────────────────────
# how_you_know: friend | colleague | classmate | mentor | client | ...
contacts:
  - name: Vikhyath
    how_you_know: colleague
    email: ksvikhyath@gmail.com
    phone: 
    company: 
    role: 
  # add more...

# ── WORK / STUDY (:Organization) ───────────────────────────
organizations:
  - name: PES University       # correct the name if needed
    your_role: Student
  # - name:
  #   your_role:

# ── PROJECTS (:Project) ────────────────────────────────────
projects:
  - name: OrbixAI
    status: in_progress        # planning | in_progress | done | on_hold
    deadline:                  # capstone due date?
    repo_url:
    description: Personal AI assistant (capstone)
  # - name:
  #   status:
  #   deadline:

# ── TASKS / TODOS (:Task) ──────────────────────────────────
tasks:
  # - title:
  #   due_date:
  #   priority:                # high | medium | low
  #   project:                 # which project (optional)

# ── TRIPS (:Trip) — planned or recent ──────────────────────
trips:
  # - destination:
  #   origin:
  #   start_date:
  #   end_date:
  #   status:                  # planned | booked | done

# ── PREFERENCES (:Preference) ──────────────────────────────
# free-form key: value pairs — anything the assistant should respect
preferences:
  ui_design: Linear/Vercel minimal — no glass/glow/gradients
  # diet:
  # cuisine:
  # communication_style:       # e.g. concise, no fluff
  # coffee:

# ── INTERESTS / HOBBIES (:Interest) ────────────────────────
interests: [coding, watching movies]                  # e.g. [coding, football, music, anime]

# ── IMPORTANT DATES (:Event) — anniversaries / recurring ───
events:
  # - title:                   # e.g. "Parents' anniversary"
  #   date:                    # YYYY-MM-DD
  #   recurrence: yearly

# ── MEDICATIONS (:Medication) — optional ───────────────────
medications: []
  # - name:
  #   dosage:
  #   schedule:                # e.g. "8am daily"
```
