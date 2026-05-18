"""NBA Q&A Streamlit UI — powered by Ollama + qwen2.5-coder:7b"""

import re
import traceback
import urllib.request
import urllib.error
import json

import pandas as pd
import streamlit as st

DATA_DIR = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b"

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NBA Q&A",
    page_icon="🏀",
    layout="wide",
)

st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stMetric {
    background: #21262d;
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    border: 1px solid #30363d;
}
[data-testid="stSidebar"] .stMetric label { color: #8b949e !important; font-size: 0.72rem !important; }
[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] { font-size: 1.05rem !important; font-weight: 600 !important; }
[data-testid="stSidebar"] hr { border-color: #21262d !important; }

/* ── Sidebar example buttons → chips ── */
[data-testid="stSidebar"] .stButton button {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 20px !important;
    color: #c9d1d9 !important;
    font-size: 0.78rem !important;
    padding: 0.35rem 0.9rem !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    background: #388bfd22 !important;
    border-color: #388bfd !important;
    color: #79c0ff !important;
}

/* ── Clear button ── */
[data-testid="stSidebar"] .stButton:last-of-type button {
    background: #21262d !important;
    border: 1px solid #f8514955 !important;
    border-radius: 8px !important;
    color: #f85149 !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] .stButton:last-of-type button:hover {
    background: #f8514922 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.2rem 0.5rem;
    margin-bottom: 0.25rem;
}

/* ── User bubble ── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #161b22;
    border: 1px solid #21262d;
}

/* ── Assistant bubble ── */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #0d1117;
    border: 1px solid #21262d;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Chat input ── */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
    border: 1px solid #30363d !important;
    background: #161b22 !important;
    color: #e6edf3 !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus { border-color: #388bfd !important; box-shadow: 0 0 0 2px #388bfd33 !important; }

/* ── Status widget ── */
[data-testid="stStatus"] {
    border-radius: 10px !important;
    border: 1px solid #21262d !important;
    background: #161b22 !important;
    font-size: 0.82rem !important;
}

/* ── Welcome hero ── */
.welcome-hero {
    text-align: center;
    padding: 4rem 2rem 2rem;
    color: #8b949e;
}
.welcome-hero h1 { font-size: 3rem; margin-bottom: 0.25rem; color: #e6edf3; font-weight: 700; }
.welcome-hero p  { font-size: 1.05rem; color: #8b949e; margin-bottom: 2.5rem; }
.chip-grid {
    display: flex; flex-wrap: wrap; gap: 0.6rem;
    justify-content: center; max-width: 680px; margin: 0 auto;
}
.chip {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 20px; padding: 0.45rem 1rem;
    font-size: 0.85rem; color: #c9d1d9; cursor: pointer;
    transition: all 0.15s ease;
}
</style>
""", unsafe_allow_html=True)

# ── Load data (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading NBA data…")
def load_tables():
    files = {
        "advanced": "advanced_clean.csv",
        "allstar": "allstar_clean.csv",
        "award_shares": "award_shares_clean.csv",
        "draft": "draft_clean.csv",
        "end_of_season": "end_of_season_clean.csv",
        "eos_voting": "eos_voting_clean.csv",
        "opp_per_100": "opp_per_100_clean.csv",
        "opp_per_game": "opp_per_game_clean.csv",
        "opp_totals": "opp_totals_clean.csv",
        "per_100": "per_100_clean.csv",
        "per_36": "per_36_clean.csv",
        "play_by_play": "play_by_play_clean.csv",
        "player_career_info": "player_career_info_clean.csv",
        "player_per_game": "player_per_game_clean.csv",
        "player_season_info": "player_season_info_clean.csv",
        "player_totals": "player_totals_clean.csv",
        "shooting": "shooting_clean.csv",
        "team_per_100": "team_per_100_clean.csv",
        "team_per_game": "team_per_game_clean.csv",
        "team_summaries": "team_summaries_clean.csv",
        "team_totals": "team_totals_clean.csv",
    }
    t = {name: pd.read_csv(f"{DATA_DIR}/{fname}") for name, fname in files.items()}

    # Pre-joined helper tables so the model never has to do joins manually
    career = t["player_career_info"]
    season_teams = t["player_season_info"][["player_id", "player", "team"]].drop_duplicates()

    # player_career_with_teams: one row per (player, team they played for)
    t["player_career_with_teams"] = career.merge(season_teams, on="player_id", how="left") \
        .rename(columns={"player_x": "player"}) \
        .drop(columns=["player_y"], errors="ignore") \
        [["player", "player_id", "from", "to", "hof", "career_length", "team"]] \
        .drop_duplicates()

    return t

TABLES = load_tables()

# ── Schema / system prompt ───────────────────────────────────────────────────
SCHEMA = """
Available pandas DataFrames (accessed as tables["name"]):

tables["player_per_game"]   — season, lg, player, player_id, team, pos, g, fg_per_game, fga_per_game, ft_per_game, fta_per_game, ast_per_game, pts_per_game, is_total_row
tables["player_totals"]     — season, lg, player, player_id, team, pos, g, fg, fga, ft, fta, ast, pts, is_total_row
tables["advanced"]          — season, lg, player, player_id, team, pos, g, is_total_row, low_sample
tables["per_100"]           — season, lg, player, player_id, age, team, pos, g, mp, is_total_row
tables["per_36"]            — season, lg, player, player_id, team, pos, g, mp, is_total_row
tables["play_by_play"]      — season, lg, player, player_id, age, team, pos, g, gs, mp, bad_pass_turnover, lost_ball_turnover, shooting_foul_committed, offensive_foul_committed, shooting_foul_drawn, points_generated_by_assists, and1, fga_blocked, pos_pct_sum
tables["shooting"]          — season, lg, player, player_id, age, team, pos, g, gs, mp, num_of_dunks, num_heaves_attempted, num_heaves_made
tables["player_career_info"]— player, player_id, from, to, hof, career_length  (no team column — use player_career_with_teams instead)
tables["player_career_with_teams"] — player, player_id, from, to, hof, career_length, team  (one row per player per team they played for; use this for any question combining career info with teams)
tables["player_season_info"]— season, lg, player, player_id, team, experience
tables["allstar"]           — player, player_id, team, season, lg, replaced
tables["award_shares"]      — season, award, player, player_id, age, winner  (award values: 'nba mvp','nba dpoy','nba roy','nba smoy','nba mip','nba clutch_poy','aba mvp','aba roy','baa roy')
tables["end_of_season"]     — season, lg, type, number_tm, player, player_id  (type: 'All-NBA','All-Defensive','All-Rookie')
tables["eos_voting"]        — season, lg, player, player_id, age
tables["draft"]             — season, lg, tm, player, player_id
tables["team_summaries"]    — season, lg, team, abbreviation, playoffs
tables["team_totals"]       — season, lg, team, playoffs
tables["team_per_game"]     — season, lg, team, playoffs
tables["team_per_100"]      — season, lg, team, abbreviation, playoffs, g, mp, fg_per_100_poss..pts_per_100_poss
tables["opp_per_100"]       — season, lg, team, abbreviation, playoffs, g, mp, opp_fg_per_100_poss..opp_pts_per_100_poss
tables["opp_per_game"]      — season, lg, team, playoffs
tables["opp_totals"]        — season, lg, team, playoffs

Key rules:
- Always filter is_total_row == False for player stats (avoids double-counting players who changed teams mid-season)
- award_shares.winner == True means the player WON that award that season
- player_career_info has NO team column — always use player_career_with_teams when team info is needed
- player_career_with_teams.hof == True means Hall of Famer
- Seasons are integers: 2024 = the 2023-24 season
- Data spans 1947 to 2026
- Teams are stored as ABBREVIATIONS only: LAL=Lakers, GSW=Warriors, BOS=Celtics, CHI=Bulls, MIA=Heat, SAS=Spurs, NYK=Knicks, PHI=76ers, DET=Pistons, HOU=Rockets, PHX=Suns, DAL=Mavericks, DEN=Nuggets, POR=TrailBlazers, SEA=SuperSonics. Never search by full city/team name.
- HOF Lakers example: tables["player_career_with_teams"][(tables["player_career_with_teams"]["hof"] == True) & (tables["player_career_with_teams"]["team"] == "LAL")][["player"]].drop_duplicates()
"""

SYSTEM_PROMPT = f"""You are an NBA data analyst. Answer questions using pandas DataFrames.

{SCHEMA}

INSTRUCTIONS:
1. Write Python/pandas code in a ```python code block to answer the question.
2. The code runs with `tables` (dict of DataFrames above) and `pd` (pandas) available.
3. Always assign the final answer to a variable called `result`.
4. After the code block, wait — the result will be shown to you, then give a plain-English answer.
5. Keep result DataFrames to ≤15 rows with .head(15) unless asked for more.
6. Round floats: .round(2)
7. For player name searches use case-insensitive .str.contains(name, case=False, na=False)
8. NEVER use df.query() — always use boolean indexing: df[df['col'] == value]
"""

# ── Ollama helpers ───────────────────────────────────────────────────────────
def ollama_chat(messages: list, stream_into=None) -> str:
    """Call Ollama. If stream_into is a st.empty(), stream tokens into it live."""
    payload = json.dumps({"model": MODEL, "messages": messages, "stream": stream_into is not None}).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            if stream_into is None:
                data = json.loads(resp.read())
                st.session_state.tokens_in  += data.get("prompt_eval_count", 0)
                st.session_state.tokens_out += data.get("eval_count", 0)
                return data["message"]["content"]
            else:
                full_text = ""
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if not line:
                        continue
                    chunk = json.loads(line)
                    full_text += chunk.get("message", {}).get("content", "")
                    stream_into.markdown(full_text + "▌")
                    if chunk.get("done"):
                        st.session_state.tokens_in  += chunk.get("prompt_eval_count", 0)
                        st.session_state.tokens_out += chunk.get("eval_count", 0)
                stream_into.markdown(full_text)
                return full_text
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach Ollama. Is `ollama serve` running?\n{e}")


def extract_code(text: str) -> str | None:
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if "tables" in code or "result" in code:
            return code
    return None


def _rewrite_query(code: str) -> str:
    """Convert df.query('col == val') to df[df['col'] == val] for simple cases."""
    def replacer(m):
        df = m.group(1)
        expr = m.group(2)
        # Handle simple: col == val, col != val, col > val, col < val
        simple = re.match(r"^(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)$", expr.strip())
        if simple:
            col, op, val = simple.group(1), simple.group(2), simple.group(3).strip()
            return f"{df}[{df}['{col}'] {op} {val}]"
        # Compound: col1 == val1 and col2 == val2
        parts = re.split(r"\s+and\s+", expr, flags=re.IGNORECASE)
        if len(parts) > 1:
            conditions = []
            for part in parts:
                s = re.match(r"^(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)$", part.strip())
                if s:
                    conditions.append(f"({df}['{s.group(1)}'] {s.group(2)} {s.group(3).strip()})")
                else:
                    return m.group(0)  # give up, return original
            return f"{df}[{' & '.join(conditions)}]"
        return m.group(0)  # can't rewrite, leave as-is (will error naturally)
    return re.sub(r"(\w+)\.query\(['\"](.+?)['\"]\)", replacer, code)


def run_code(code: str) -> tuple[str, pd.DataFrame | None]:
    """Returns (text_output, dataframe_or_None)."""
    code = _rewrite_query(code)
    namespace = {"tables": TABLES, "pd": pd}
    try:
        exec(code, namespace)
        result = namespace.get("result", None)
        if result is None:
            return "No `result` variable was set.", None
        if isinstance(result, pd.DataFrame):
            if result.empty:
                return "(empty DataFrame)", None
            return result.round(2).to_string(index=False), result.round(2)
        if isinstance(result, pd.Series):
            return result.round(2).to_string(), None
        if isinstance(result, float):
            return f"{result:.2f}", None
        return str(result), None
    except Exception:
        return f"ERROR:\n{traceback.format_exc()}", None


def agent_turn(question: str, history: list, status_box, stream_box) -> tuple[str, list, pd.DataFrame | None]:
    """One agent turn. Returns (answer, updated_history, optional_dataframe)."""
    history = history + [{"role": "user", "content": question}]
    result_df = None

    for attempt in range(3):
        label = "Writing query…" if attempt == 0 else f"Retrying… (attempt {attempt + 1}/3)"
        status_box.status(label, state="running")
        response_text = ollama_chat(history)
        history = history + [{"role": "assistant", "content": response_text}]

        code = extract_code(response_text)
        if code is None:
            # Direct answer with no code — stream it
            stream_box.markdown(response_text)
            return response_text, history, None

        status_box.status("Running query…", state="running")
        output, result_df = run_code(code)

        if output.startswith("ERROR") or output == "(empty DataFrame)":
            result_df = None
            if attempt < 2:
                short_reason = output.splitlines()[1] if output.startswith("ERROR") else "empty result"
                status_box.status(f"Query failed ({short_reason[:60]}) — retrying…", state="running")
                history = history + [{
                    "role": "user",
                    "content": f"The query returned this problem:\n```\n{output}\n```\nRewrite the code using boolean indexing (never .query()). Assign result to `result`."
                }]
                continue
            else:
                msg = f"Couldn't retrieve data after 3 attempts.\n\nLast error:\n```\n{output}\n```"
                stream_box.markdown(msg)
                return msg, history, None

        feedback = (
            f"Query result:\n```\n{output}\n```\n"
            "Give a concise, conversational answer based only on the data above. "
            "Do not invent numbers not shown in the result."
        )
        history = history + [{"role": "user", "content": feedback}]
        status_box.status("Answering…", state="running")
        final = ollama_chat(history, stream_into=stream_box)
        history = history + [{"role": "assistant", "content": final}]
        return final, history, result_df

    return "I wasn't able to answer that after multiple attempts.", history, None


# ── Session state (must init before sidebar reads it) ────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "tokens_in" not in st.session_state:
    st.session_state.tokens_in = 0
if "tokens_out" not in st.session_state:
    st.session_state.tokens_out = 0

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem;">
        <div style="font-size:1.6rem; font-weight:700; color:#e6edf3; letter-spacing:-0.5px;">🏀 NBA Q&A</div>
        <div style="font-size:0.72rem; color:#8b949e; margin-top:2px;">qwen2.5-coder:7b · local</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Dataset**")
    total_rows = sum(len(df) for df in TABLES.values())
    c1, c2 = st.columns(2)
    c1.metric("Seasons", "1947–2026")
    c2.metric("Rows", f"{total_rows:,}")

    st.divider()
    st.markdown("**Token usage**")
    total_tokens = st.session_state.tokens_in + st.session_state.tokens_out
    t1, t2 = st.columns(2)
    t1.metric("In", f"{st.session_state.tokens_in:,}")
    t2.metric("Out", f"{st.session_state.tokens_out:,}")
    st.caption(f"Session total: **{total_tokens:,}**")

    st.divider()
    st.markdown("**Try asking…**")
    examples = [
        "Who won the most MVP awards?",
        "Top 5 scorers per game in 2024?",
        "Which teams made the playoffs most often?",
        "Who has the most All-Star appearances?",
        "Most dunks in a single season?",
        "Which Hall of Famers played for the Lakers?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state.pending_question = ex

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        st.session_state.tokens_in = 0
        st.session_state.tokens_out = 0
        st.rerun()

# ── Main chat area ────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-hero">
        <h1>🏀</h1>
        <h1 style="font-size:2rem; margin-top:0;">Ask anything about NBA history</h1>
        <p>80 years of stats, awards, drafts, and play-by-play data — 1947 to 2026</p>
        <div class="chip-grid">
            <span class="chip">Who won the most MVPs?</span>
            <span class="chip">Top scorers in 2024</span>
            <span class="chip">Most dunks in a season</span>
            <span class="chip">HOF Lakers players</span>
            <span class="chip">Most All-Star appearances</span>
            <span class="chip">Teams with most playoffs</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🏀" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg.get("dataframe") is not None:
            st.dataframe(msg["dataframe"], use_container_width=True, hide_index=True)

# Handle sidebar example button clicks
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
else:
    question = st.chat_input("Ask a question about NBA history…")

if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Run agent and stream status
    with st.chat_message("assistant", avatar="🏀"):
        status = st.empty()
        stream = st.empty()
        try:
            answer, new_history, result_df = agent_turn(
                question, st.session_state.history, status, stream
            )
            status.empty()
            if result_df is not None:
                st.dataframe(result_df, use_container_width=True, hide_index=True)
        except RuntimeError as e:
            status.empty()
            answer = f"⚠️ {e}"
            result_df = None
            st.error(answer)

    st.session_state.history = new_history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "dataframe": result_df,
    })
