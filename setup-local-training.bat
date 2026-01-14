@echo off
echo Setting up FREE Local Mage RL Training
echo Cost: $0/day (uses your local machine)

REM Create local directories
echo Setting up local directories...
mkdir local-training\trajectories 2>nul
mkdir local-training\models 2>nul
mkdir local-training\logs 2>nul
mkdir local-training\stats 2>nul

REM Create environment file
if not exist .env (
    echo Creating environment file...
    (
    echo # Local-only training defaults (no external API required^)
    echo.
    echo # Optional: If you later re-enable text embeddings
    echo # OPENAI_API_KEY=your_openai_api_key_here
    echo.
    echo # Use a fixed deck pool for the Pauper subset milestone
    echo # Paths are resolved relative to the container working directory if running in Docker.
    echo DECK_LIST_FILE=Mage.Server.Plugins/Mage.Player.AIRL/src/mage/player/ai/decks/PauperSubset/decklist.txt
    ) > .env
    echo INFO: Created .env with local-only defaults
)

echo.
echo Local training setup complete!
echo.
echo Next Steps:
echo   1. (Optional) Edit .env and adjust DECK_LIST_FILE
echo   2. Run training (CPU worker + GPU learner): docker-compose -f docker-compose-gpu.yml up
echo   3. Run a quick benchmark: set MODE=benchmark and GAMES_PER_MATCHUP=20
echo.
echo Estimated Costs:
echo   - Hardware: $0 (uses your existing computer)
echo   - Electricity: ~$1-3/day
echo   - Total: ~$1-3/day

pause 