# 🏀 NBA Q&A

An AI-powered NBA research tool that answers natural language questions using 80 years of basketball data (1947–2026). Runs fully locally via [Ollama](https://ollama.com) — no API key required.

## Demo

Ask questions like:
- *"Who won the most MVP awards?"*
- *"Top 5 scorers per game in 2024?"*
- *"Which Hall of Famers played for the Lakers?"*
- *"Most dunks in a single season?"*
- *"Which teams made the playoffs most often?"*

The agent writes pandas code against the dataset, executes it, and gives a conversational answer.

## Setup

**1. Install dependencies**

```bash
pip install streamlit pandas
```

**2. Install Ollama and pull the model**

Download Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull qwen2.5-coder:7b
```

**3. Start Ollama**

```bash
ollama serve
```

**4. Run the UI**

```bash
streamlit run nba_ui.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

Or use the terminal agent instead:

```bash
python3 nba_agent.py
```

## Dataset

21 CSV tables sourced from [Basketball-Reference](https://www.basketball-reference.com), covering NBA/ABA/BAA seasons from 1947 to 2026.

| Table | Description |
|---|---|
| `player_per_game` | Points, assists, FG per game by player/season |
| `player_totals` | Season totals (FG, FT, AST, PTS, etc.) |
| `advanced` | Advanced stats per season |
| `per_100` / `per_36` | Pace-adjusted stats |
| `play_by_play` | Turnovers, fouls drawn, and1s, blocked shots |
| `shooting` | Dunks, heaves |
| `player_career_info` | Career span, Hall of Fame status |
| `player_season_info` | Team and experience by season |
| `allstar` | All-Star game appearances |
| `award_shares` | MVP, DPOY, ROY, SMOY, MIP voting and winners |
| `end_of_season` | All-NBA, All-Defensive, All-Rookie teams |
| `draft` | Draft picks by season |
| `team_summaries` | Team records and playoff appearances |
| `team_per_game` / `team_per_100` / `team_totals` | Team offensive stats |
| `opp_per_game` / `opp_per_100` / `opp_totals` | Team defensive stats |

## How it works

1. All CSVs are loaded into pandas DataFrames at startup
2. Your question is sent to `qwen2.5-coder:7b` running locally via Ollama
3. The model generates pandas code to query the data
4. The code runs and the result is fed back to the model
5. The model returns a plain-English answer, streamed token by token
6. If the query errors, the agent retries up to 3 times with corrected code

## Project structure

```
├── nba_ui.py          # Streamlit web UI
├── nba_agent.py       # Terminal chat agent
├── *_clean.csv        # Cleaned NBA datasets
└── README.md
```
