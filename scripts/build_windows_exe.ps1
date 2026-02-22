$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

python -m pip install --upgrade pyinstaller

if (Test-Path "$root\build") {
  Remove-Item "$root\build" -Recurse -Force
}
if (Test-Path "$root\dist") {
  Remove-Item "$root\dist" -Recurse -Force
}

python -m PyInstaller `
  --noconfirm `
  --clean `
  --name FaceAttendanceDashboard `
  --onefile `
  --add-data "web;web" `
  --collect-all mediapipe `
  dashboard_launcher.py

Write-Host ""
Write-Host "Windows EXE built:"
Write-Host "$root\dist\FaceAttendanceDashboard.exe"
