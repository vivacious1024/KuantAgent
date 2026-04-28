param(
    [string]$BenchmarkDir = "f:\毕设\Code\KuantAgent\benchmark\1h\btc",
    [string]$Timeframe = "1h",
    [int]$WindowSize = 45,
    [int]$FutureHorizon = 3,
    [string]$OutputDir = ""
)

$scriptPath = Join-Path $PSScriptRoot "run_pure_algo_experiment.py"

if (-not (Test-Path $scriptPath)) {
    throw "未找到实验脚本: $scriptPath"
}

$pythonExe = $null
$pythonArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = "py"
    $pythonArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = "python"
    $pythonArgs = @()
} else {
    throw "未找到 Python 解释器。请先确认 py 或 python 可用。"
}

$arguments = @(
    $scriptPath,
    "--benchmark-dir", $BenchmarkDir,
    "--timeframe", $Timeframe,
    "--window-size", "$WindowSize",
    "--future-horizon", "$FutureHorizon"
)

if ($OutputDir -ne "") {
    $arguments += @("--output-dir", $OutputDir)
}

Write-Host "开始运行纯算法层实验..." -ForegroundColor Cyan
Write-Host "BenchmarkDir: $BenchmarkDir"
Write-Host "Timeframe: $Timeframe"
Write-Host "WindowSize: $WindowSize"
Write-Host "FutureHorizon: $FutureHorizon"
Write-Host ""

& $pythonExe @pythonArgs @arguments
