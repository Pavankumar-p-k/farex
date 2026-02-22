$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$mobileDir = Join-Path $root "mobile_dashboard"
$androidStudioJbr = "C:\Program Files\Android\Android Studio\jbr"
$androidSdk = Join-Path $env:LOCALAPPDATA "Android\Sdk"

if (Test-Path $androidStudioJbr) {
  $env:JAVA_HOME = $androidStudioJbr
  $env:Path = "$env:JAVA_HOME\bin;$env:Path"
}

if (Test-Path $androidSdk) {
  $env:ANDROID_HOME = $androidSdk
  $env:ANDROID_SDK_ROOT = $androidSdk
  $env:Path = "$env:ANDROID_HOME\platform-tools;$env:Path"
}

Set-Location $mobileDir
npm install

if (-not (Test-Path (Join-Path $mobileDir "android"))) {
  npx cap add android
}

npx cap sync android

Set-Location (Join-Path $mobileDir "android")
.\gradlew.bat assembleDebug

$apkPath = Join-Path $mobileDir "android\app\build\outputs\apk\debug\app-debug.apk"
Write-Host ""
Write-Host "Android APK built:"
Write-Host $apkPath
