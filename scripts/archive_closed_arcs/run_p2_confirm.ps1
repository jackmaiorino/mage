# STEP-2 CONFIRM: firm up the "true combo-ready reach ~0%" claim at N=256 (Codex #17).
# Greedy eval, gate+search+diag ON. Aggregates the in-engine funnel counters (summed across JVMs):
#   libLands==0 decisions, libLands<=1, libEmpty, board>=2, both(combo-ready), total; + search CALLS/FOUND.
# 0 combo-ready over 256 games -> 95% upper bound ~1.2% (cleanly fails the >=2% gate).
Set-Location "C:\Users\Jack\IdeaProjects\mage"
$ErrorActionPreference = "Continue"
$out  = "local-training/p2_confirm_RESULT.log"
$reg  = "local-training/_brew_win_registry.json"
$md   = "Mage.Server.Plugins\Mage.Player.AIRL\src\mage\player\ai\rl\profiles\Pauper-Spy-Combo-Value\models"
$best = "E:\mage-training\backups\league_step1_20260620\model_best.pt"

function Kill-Train { Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'java.exe' -or ($_.Name -match 'python' -and $_.CommandLine -match 'gpu_service_host|run_local_pbt') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }

"=== P2 CONFIRM (N=256) $(Get-Date) ===" | Out-File $out
Kill-Train; Start-Sleep -Seconds 3
if (Test-Path "$md\onnx") { Remove-Item "$md\onnx" -Recurse -Force }
Copy-Item $best "$md\model_latest.pt" -Force; Copy-Item $best "$md\model.pt" -Force
foreach($k in 'RL_ZONE_COUNT_FEATURES_ENABLE','RL_LIBRARY_COUNT_FEATURES_ENABLE','VALUE_USE_SEPARATE_CRITIC_ENCODER','WORLD_MODEL_DIM'){ Remove-Item "Env:\$k" -ErrorAction SilentlyContinue }
$env:SEARCH_OP_ENABLE="0"; $env:USE_TRT_INFERENCE="0"
$env:MODEL_D_MODEL="128"; $env:MODEL_NUM_LAYERS="2"; $env:MODEL_NHEAD="4"; $env:MODEL_DIM_FEEDFORWARD="512"
$env:RL_ONLINE_PREFIX_DIAG="1"
$env:RL_ONLINE_PREFIX_SEARCH_ENABLE="1"
$env:RL_ONLINE_PREFIX_COMBO_READY_GATE="1"
$env:RL_ONLINE_PREFIX_COMBO_READY_MIN_CREATURES="2"
$env:RL_ONLINE_PREFIX_GENERIC_BRANCH_ORDER="1"
$env:RL_ONLINE_PREFIX_AUTOPILOT_ENABLE="1"
$env:RL_ONLINE_PREFIX_SEARCH_MAX_DEPTH="12"
$env:RL_ONLINE_PREFIX_SEARCH_MAX_NODES="400"
$env:RL_ONLINE_PREFIX_SEARCH_TOP_K="4"
$env:RL_ONLINE_PREFIX_SEARCH_MAX_ACTIVATIONS="20"
$env:RL_ONLINE_PREFIX_SEARCH_TOTAL_TIMEOUT_MS="8000"
$env:RL_ONLINE_PREFIX_SEARCH_BRANCH_TIMEOUT_MS="3000"

Remove-Item -Recurse -Force "local-training/local_pbt/cp7_eval_sweeps/p2_confirm" -ErrorAction SilentlyContinue
py -3.12 scripts/run_cp7_eval_sweep.py --registry $reg --profiles Pauper-Spy-Combo-Value `
  --opponents grixis --skill 1 --games-per-matchup 256 --parallel 8 --games-per-job 16 `
  --eval-game-logging --skip-compile --replay-seed-base 5151 --run-id "p2_confirm" 2>&1 |
  Select-String "wr=" | ForEach-Object { $_.Line } | Out-File $out -Append

# aggregate: sum the last OPDIAG-FUNNEL per job log + winrate + search calls
"--- aggregate ---" | Out-File $out -Append
py -3.12 -c "
import glob, re, os
lp = glob.glob('local-training/local_pbt/cp7_eval_sweeps/p2_confirm/logs/*.log')
fr = re.compile(r'total=(\d+) libLands0=(\d+) libLands<=1=(\d+) libEmpty=(\d+) board>=2=(\d+) both=(\d+)')
cr = re.compile(r'calls=(\d+) found=(\d+)')
tot=l0=l1=le=b2=bo=calls=found=0
for f in lp:
    if os.path.basename(f)=='gpu_service.log': continue
    lastf=None; lastc=(0,0)
    for line in open(f, encoding='utf-8', errors='replace'):
        m=fr.search(line)
        if m: lastf=tuple(int(x) for x in m.groups())
        c=cr.search(line)
        if c:
            v=tuple(int(x) for x in c.groups())
            if v[0]>=lastc[0]: lastc=v
    if lastf:
        tot+=lastf[0]; l0+=lastf[1]; l1+=lastf[2]; le+=lastf[3]; b2+=lastf[4]; bo+=lastf[5]
    calls+=lastc[0]; found+=lastc[1]
print(f'decisions_total={tot}')
print(f'libLands==0 decisions = {l0}   libLands<=1 = {l1}   libEmpty = {le}')
print(f'board>=2 decisions   = {b2} ({(b2/tot if tot else 0):.0%})')
print(f'combo-ready (both)   = {bo}   -> reach = {(bo/tot if tot else 0):.4%} of decisions')
print(f'online-prefix search CALLS={calls} FOUND={found}')
print('VERDICT: combo-ready reach=0 over N=256 -> 95%% upper bound ~1.2%%; finish-search dead.' if bo==0 else f'NOTE: {bo} combo-ready decisions seen -- revisit.')
" 2>&1 | Out-File $out -Append
"=== P2 CONFIRM DONE $(Get-Date) ===" | Out-File $out -Append
