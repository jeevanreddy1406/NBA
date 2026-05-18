#!/usr/bin/env python3
"""NBA Q&A Agent — answers natural language questions using local Ollama model."""

import os
import re
import sys
import traceback
import pandas as pd
import urllib.request
import urllib.error
import json

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:7b"

# Load all CSVs once at startup
print("Loading NBA data...", flush=True)
TABLES = {
    "advanced": pd.read_csv(f"{DATA_DIR}/advanced_clean.csv"),
    "allstar": pd.read_csv(f"{DATA_DIR}/allstar_clean.csv"),
    "award_shares": pd.read_csv(f"{DATA_DIR}/award_shares_clean.csv"),
    "draft": pd.read_csv(f"{DATA_DIR}/draft_clean.csv"),
    "end_of_season": pd.read_csv(f"{DATA_DIR}/end_of_season_clean.csv"),
    "eos_voting": pd.read_csv(f"{DATA_DIR}/eos_voting_clean.csv"),
    "opp_per_100": pd.read_csv(f"{DATA_DIR}/opp_per_100_clean.csv"),
    "opp_per_game": pd.read_csv(f"{DATA_DIR}/opp_per_game_clean.csv"),
    "opp_totals": pd.read_csv(f"{DATA_DIR}/opp_totals_clean.csv"),
    "per_100": pd.read_csv(f"{DATA_DIR}/per_100_clean.csv"),
    "per_36": pd.read_csv(f"{DATA_DIR}/per_36_clean.csv"),
    "play_by_play": pd.read_csv(f"{DATA_DIR}/play_by_play_clean.csv"),
    "player_career_info": pd.read_csv(f"{DATA_DIR}/player_career_info_clean.csv"),
    "player_per_game": pd.read_csv(f"{DATA_DIR}/player_per_game_clean.csv"),
    "player_season_info": pd.read_csv(f"{DATA_DIR}/player_season_info_clean.csv"),
    "player_totals": pd.read_csv(f"{DATA_DIR}/player_totals_clean.csv"),
    "shooting": pd.read_csv(f"{DATA_DIR}/shooting_clean.csv"),
    "team_per_100": pd.read_csv(f"{DATA_DIR}/team_per_100_clean.csv"),
    "team_per_game": pd.read_csv(f"{DATA_DIR}/team_per_game_clean.csv"),
    "team_summaries": pd.read_csv(f"{DATA_DIR}/team_summaries_clean.csv"),
    "team_totals": pd.read_csv(f"{DATA_DIR}/team_totals_clean.csv"),
}
print(f"Loaded {len(TABLES)} tables. Data covers seasons 1947–2026.\n", flush=True)

SCHEMA = """
Available pandas DataFrames (accessed as tables["name"]):

tables["player_per_game"]   — season, lg, player, player_id, team, pos, g, fg_per_game, fga_per_game, ft_per_game, fta_per_game, ast_per_game, pts_per_game, is_total_row
tables["player_totals"]     — season, lg, player, player_id, team, pos, g, fg, fga, ft, fta, ast, pts, is_total_row
tables["advanced"]          — season, lg, player, player_id, team, pos, g, is_total_row, low_sample
tables["per_100"]           — season, lg, player, player_id, age, team, pos, g, mp, is_total_row
tables["per_36"]            — season, lg, player, player_id, team, pos, g, mp, is_total_row
tables["play_by_play"]      — season, lg, player, player_id, age, team, pos, g, gs, mp, bad_pass_turnover, lost_ball_turnover, shooting_foul_committed, offensive_foul_committed, shooting_foul_drawn, points_generated_by_assists, and1, fga_blocked, pos_pct_sum
tables["shooting"]          — season, lg, player, player_id, age, team, pos, g, gs, mp, num_of_dunks, num_heaves_attempted, num_heaves_made
tables["player_career_info"]— player, player_id, from, to, hof, career_length
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
- Always filter is_total_row == False for player stats (avoids double-counting players who changed teams)
- award_shares.winner == True means the player WON that award
- player_career_info.hof == True means Hall of Famer
- Seasons are integers: 2024 = the 2023-24 season
- Data spans 1947 to 2026
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
"""


def ollama_chat(messages: list) -> str:
    """Send messages to local Ollama and return the response text."""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data["message"]["content"]
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Is 'ollama serve' running?\nError: {e}"
        )


def extract_code(text: str) -> str | None:
    """Extract the first ```python ... ``` block from model output."""
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: bare ``` block
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "tables" in code or "result" in code:
            return code
    return None


def run_code(code: str) -> str:
    """Execute code with tables and pd in scope; return string output."""
    namespace = {"tables": TABLES, "pd": pd}
    try:
        exec(code, namespace)
        result = namespace.get("result", None)
        if result is None:
            return "No `result` variable was set."
        if isinstance(result, pd.DataFrame):
            return "(empty DataFrame)" if result.empty else result.round(2).to_string(index=False)
        if isinstance(result, pd.Series):
            return result.round(2).to_string()
        if isinstance(result, float):
            return f"{result:.2f}"
        return str(result)
    except Exception:
        return f"ERROR:\n{traceback.format_exc()}"


def chat(question: str, history: list) -> tuple[str, list]:
    """Run one turn of the conversation; returns (answer_text, updated_history)."""
    history = history + [{"role": "user", "content": question}]

    # Up to 3 attempts if code errors out
    for attempt in range(3):
        response_text = ollama_chat(history)
        history = history + [{"role": "assistant", "content": response_text}]

        code = extract_code(response_text)
        if code is None:
            # No code block — model gave a direct answer
            return response_text, history

        print(f"\n[Querying data...]", flush=True)
        output = run_code(code)

        if output.startswith("ERROR") or output == "(empty DataFrame)":
            if attempt < 2:
                reason = "an error" if output.startswith("ERROR") else "an empty result"
                history = history + [{"role": "user", "content": f"The query returned {reason}:\n```\n{output}\n```\nPlease rewrite the Python code to fix this and try again."}]
                continue
            else:
                return f"I wasn't able to retrieve data for that question (got: {output}).", history

        # Feed result back so model can interpret it
        feedback = f"Query result:\n```\n{output}\n```\nNow give a clear, conversational answer to the user's question based on this data. Do not make up numbers — use only what's in the result above."
        history = history + [{"role": "user", "content": feedback}]

        # Get the final conversational answer
        final = ollama_chat(history)
        history = history + [{"role": "assistant", "content": final}]
        return final, history

    return "I wasn't able to answer that after multiple attempts.", history


def main():
    print("=" * 60)
    print("NBA Q&A Agent  (powered by Ollama + qwen2.5-coder:7b)")
    print("Ask any question about NBA history (1947–2026).")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    # Prepend system message
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            answer, history = chat(question, history)
            print(f"\nAgent: {answer}\n")
        except RuntimeError as e:
            print(f"\n[Error: {e}]\n")
        except Exception as e:
            print(f"\n[Unexpected error: {e}]\n")


if __name__ == "__main__":
    main()
