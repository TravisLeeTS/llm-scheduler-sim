@echo off
cd /d C:\Users\tings\llm_scheduler_sim
echo Starting v4 stress test loop...
:loop
python scripts/run_one_v4.py
if %ERRORLEVEL% EQU 0 (
    echo Completed one config, continuing...
    goto loop
) else (
    echo Error or complete, checking...
    python -c "import pandas as pd; df=pd.read_csv('stress_test_v4_results/step1_grid_search.csv'); print('Total:', len(df), '/ 192')"
    if %ERRORLEVEL% EQU 0 goto loop
)
echo Done!
pause
