@echo off
REM Comprehensive Stress Test Runner
REM Runs Step 1 (Grid Search) and Step 2 (Method Comparison)
REM Uses cmd.exe for stability on long-running tasks

echo ============================================================
echo Comprehensive Stress Test - Starting at %date% %time%
echo ============================================================
echo.

cd /d %~dp0\..
echo Working directory: %cd%
echo.

REM Clean up old results (optional)
REM if exist stress_test_final rmdir /s /q stress_test_final

echo Starting Step 1: Grid Search
echo ============================================================
python scripts\step1_grid_search.py
if %errorlevel% neq 0 (
    echo Step 1 failed with error code %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo Starting Step 2: Method Comparison
echo ============================================================
python scripts\step2_comparison.py
if %errorlevel% neq 0 (
    echo Step 2 failed with error code %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo ============================================================
echo All tests completed at %date% %time%
echo Results saved to: stress_test_final\
echo ============================================================
pause
