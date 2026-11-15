# Deploy audio-sentiment-analysis + models_final to a Hugging Face Docker Space.
# Prerequisites: hf auth login
# Usage:  powershell -ExecutionPolicy Bypass -File scripts/deploy-hf-space.ps1
# Optional:  -SpaceId "YourHFUser/voice-emotion"

param(
  [string]$SpaceId = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$AppDir = Join-Path $Root "audio-sentiment-analysis"
$Models = Join-Path $AppDir "models_final"

Write-Host "Checking Hugging Face login..."
$who = hf auth whoami 2>&1 | Out-String
if ($LASTEXITCODE -ne 0 -or $who -match "Not logged in") {
  Write-Host "Not logged in. Run:  hf auth login"
  Write-Host "Then re-run this script."
  exit 1
}
Write-Host $who.Trim()

if (-not $SpaceId) {
  $user = $null
  try {
    $json = hf auth whoami --format json 2>$null | ConvertFrom-Json
    if ($json.user) { $user = $json.user }
    elseif ($json.name) { $user = $json.name }
    elseif ($json.fullname) { $user = $json.fullname }
  } catch {}
  if (-not $user) {
    # Fallback: first non-empty line from whoami
    $user = ($who -split "`n" | ForEach-Object { $_.Trim() } | Where-Object { $_ -and $_ -notmatch "hint|error|token" } | Select-Object -First 1)
  }
  if (-not $user) {
    Write-Host "Could not detect HF username. Pass -SpaceId 'YourUser/voice-emotion'"
    exit 1
  }
  $SpaceId = "$user/voice-emotion"
}

if (-not (Test-Path $Models)) {
  Write-Host "Missing models_final at $Models - demo cannot score without weights."
  exit 1
}

$Staging = Join-Path $env:TEMP "trex-voice-emotion-space"
if (Test-Path $Staging) { Remove-Item -Recurse -Force $Staging }
New-Item -ItemType Directory -Path $Staging | Out-Null

Write-Host "Staging Space files -> $Staging"
robocopy $AppDir $Staging /E /XD .venv __pycache__ data uploads results notebooks figures .git /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
Copy-Item (Join-Path $AppDir "README_SPACE.md") (Join-Path $Staging "README.md") -Force

Write-Host "Creating Space $SpaceId (docker) if needed..."
hf repos create $SpaceId --type space --sdk docker --public --exist-ok
if ($LASTEXITCODE -ne 0) {
  Write-Host "Space create returned exit $LASTEXITCODE (may already exist); continuing upload..."
}

Write-Host "Uploading (includes models_final, ~2.7GB - may take a while)..."
hf upload $SpaceId $Staging . --repo-type space --commit-message "Deploy voice emotion demo"
if ($LASTEXITCODE -ne 0) {
  Write-Host "Upload failed with exit $LASTEXITCODE"
  exit $LASTEXITCODE
}

$AppUrl = "https://$($SpaceId.Replace('/','-').ToLower()).hf.space"
Write-Host ""
Write-Host "Upload started/finished. When the Space build is green, open:"
Write-Host "  Space page: https://huggingface.co/spaces/$SpaceId"
Write-Host "  App URL:    $AppUrl"
Write-Host ""
Write-Host "Then set window.EMOTION_DEMO_URL in demo-config.js to:"
Write-Host "  $AppUrl"
Write-Host "Commit and push so GitHub Pages picks up the link."
