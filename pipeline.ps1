# pipeline.ps1
# 用法（任意目录均可）：
#   powershell -ExecutionPolicy Bypass -File "C:\Users\Inuyasha\Desktop\AI\AI_Inv\World_Model\pipeline.ps1"
# 或在 World_Model 下：
#   .\pipeline.ps1

$ErrorActionPreference = "Stop"

# 以本脚本所在目录为锚点，定位到 scripts
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$SCRIPTS = Join-Path $ROOT "scripts"

if (!(Test-Path $SCRIPTS)) {
  Write-Host "[FATAL] scripts dir not found: $SCRIPTS"
  exit 1
}

function Run-Step($name, $cmd) {
  Write-Host "`n==================== $name ===================="
  Write-Host $cmd
  iex $cmd
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[FATAL] Step failed: $name (exit=$LASTEXITCODE)"
    exit $LASTEXITCODE
  }
}

Push-Location $SCRIPTS
try {
  # 你可以按需注释/开启某些步骤
#  Run-Step "DeepResearch (50 tasks, concurrent=5)" "python run_people_deep_research.py"
  Run-Step "Score (gpt-5.2, concurrent=20)"        "python score_people.py"
  Run-Step "Profile JSON (gpt-5.2, concurrent=20)" "python extract_profiles.py"
  Run-Step "Merge to person dict"                 "python merge_people.py"

  Write-Host "`n[OK] Pipeline finished."
}
finally {
  Pop-Location
}
