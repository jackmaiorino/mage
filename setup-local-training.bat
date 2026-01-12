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
    echo # OpenAI API Key (required for card embeddings^)
    echo OPENAI_API_KEY=your_openai_api_key_here
    echo.
    echo # Optional: Reduce API usage by caching embeddings
    echo CACHE_EMBEDDINGS=true
    echo MAX_API_CALLS_PER_HOUR=1000
    ) > .env
    echo WARNING: Please edit .env file and add your OpenAI API key
)

echo.
echo Local training setup complete!
echo.
echo Next Steps:
echo   1. Edit .env file with your OpenAI API key
echo   2. Run: docker-compose -f docker-compose-minimal.yml up
echo   3. Monitor: http://localhost:8080
echo.
echo Estimated Costs:
echo   - Hardware: $0 (uses your existing computer)
echo   - Electricity: ~$1-3/day
echo   - OpenAI API: ~$0.50-2/day (with caching)
echo   - Total: ~$1.50-5/day (96%% cheaper than cloud!)

pause 