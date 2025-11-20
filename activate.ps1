# Activate Virtual Environment for LLM Scheduler Simulation
# Usage: .\activate.ps1

Write-Host "🚀 Activating LLM Scheduler Simulation Environment..." -ForegroundColor Green
Write-Host ""

# Activate the virtual environment
.\venv\Scripts\Activate.ps1

Write-Host "✅ Virtual environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Project: C:\Users\tings\llm_scheduler_sim" -ForegroundColor Cyan
Write-Host "🐍 Python:" (python --version) -ForegroundColor Cyan
Write-Host ""
Write-Host "📝 Quick Commands:" -ForegroundColor Yellow
Write-Host "  Compare schedulers:    python scripts/run_mb_dynamic.py --compare --num-requests 2000"
Write-Host "  Run single test:       python scripts/run_mb_dynamic.py --scheduler multi_bin_dynamic --num-requests 1000"
Write-Host "  K_BINS sensitivity:    python scripts/run_mb_dynamic.py --k-bins-sensitivity --num-requests 1500"
Write-Host "  Deactivate:            deactivate"
Write-Host ""
