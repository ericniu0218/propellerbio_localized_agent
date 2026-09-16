# Installs Bedrock Private AI with one command, no Python install or venv
# needed. See INSTALL.md for the full plain-language walkthrough -- this
# comment block is for the curious, not required reading to use the script.
$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/Bedrock-Technical-Consulting/bedrock-private-ai.git"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "Installing uv (a small tool that manages Python for you)..."
    powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "Installing Bedrock Private AI..."
uv tool install --force "git+$RepoUrl"

Write-Host ""
Write-Host "Done! Start it by typing:  bedrock"
Write-Host "If PowerShell says that command isn't found, close this window,"
Write-Host "open a new PowerShell window, and try again -- that's normal the first time."
