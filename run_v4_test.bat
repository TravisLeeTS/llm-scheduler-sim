@echo off
cd /d C:\Users\tings\llm_scheduler_sim
echo ===============================================
echo V4 Stress Test - Step 1: Grid Search
echo ===============================================
python scripts/run_v4_step1.py
if errorlevel 1 (
    echo Step 1 encountered an error, but continuing...
)

echo.
echo ===============================================
echo V4 Stress Test - Step 2: Method Comparison
echo ===============================================
python scripts/run_v4_step2.py
if errorlevel 1 (
    echo Step 2 encountered an error.
)

echo.
echo ===============================================
echo All steps completed!
echo ===============================================
pause
