# MTGRL Game Log Formats

`GameLogger` supports three formats:

- `GAME_LOG_FORMAT=compact`: action-by-action trace for manual reading.
- `GAME_LOG_FORMAT=full`: original verbose format with full state and all options. This remains the default for direct runs and legacy parser workflows.
- `GAME_LOG_FORMAT=both`: compact trace followed by the original full block for each decision.

`GAME_LOG_STYLE` is accepted as an alias, and `GAME_LOG_COMPACT=1` also selects compact mode when `GAME_LOG_FORMAT` is not set.

## Compact Trace

Compact logs keep the existing `DECISION #`, `VALUE SCORE:`, and `SELECTED:` markers, but collapse the state and options into a few lines:

```text
DECISION #48 - Turn 3 (PlayerRL1 turn), Postcombat Main (TARGET_PICK 0 min=1 max=1) - PlayerRL1
  SELECTED[0] p=0.2201 value=-0.333740: Clockwork Percussionist (SelfPlay)
  STATE: stack=1 items top=[0] stack ability (...) || PlayerRL1 L17 H5[Sagu Wildling; ...] B2[Forest,tapped; Quirion Ranger] G0 X0 || SelfPlay L20 H4 B6[Mountain,tapped; ...] G0 X0
  TOP: n=4 | *[0] 0.2201 Clockwork Percussionist (SelfPlay) | [1] 0.3169 Goblin Tomb Raider (SelfPlay) | [3] 0.3819 Voldaren Epicure (SelfPlay)
VALUE SCORE: -0.333740
SELECTED: Clockwork Percussionist (SelfPlay)
```

Zone abbreviations:

- `H`: hand
- `B`: battlefield/permanents
- `G`: graveyard
- `X`: exile

For hidden hands, compact mode writes counts such as `H4`. For visible zones, it writes a count and truncated card list.

## Common Commands

Enable compact logs for an ad hoc local run:

```powershell
$env:GAME_LOGGING="1"
$env:GAME_LOG_FREQUENCY="1"
$env:GAME_LOG_FORMAT="compact"
```

Run an eval sweep with compact game logs:

```powershell
.\.mtgrl_venv\Scripts\python.exe .\scripts\run_cp7_eval_sweep.py --eval-game-logging --game-log-format compact
```

Autonomous local cycles now default logged training games to compact unless `GAME_LOG_FORMAT` is already set in the environment. Use `GAME_LOG_FORMAT=full` to force the old output.

View an existing verbose log as an action trace:

```powershell
.\.mtgrl_venv\Scripts\python.exe .\scripts\view_game_log_trace.py .\path\to\game_*.txt
```

View the newest logged game from a directory:

```powershell
.\.mtgrl_venv\Scripts\python.exe .\scripts\view_game_log_trace.py .\Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\logs\games\training --latest 1
```

## Tuning

- `GAME_LOG_COMPACT_ZONE_CHARS`: max card-list characters per compact zone, default `96`.
- `GAME_LOG_COMPACT_ACTION_CHARS`: max action text characters, default `180`.
- `GAME_LOG_COMPACT_TOP_OPTIONS`: selected option plus top policy alternatives, default `5`.
