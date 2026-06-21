param([string]$RepoPath = ".")

Set-Location $RepoPath

if (-not (Test-Path ".git")) {
    Write-Host "Not a git repository: $RepoPath" -ForegroundColor Red
    exit 1
}

$branch = git rev-parse --abbrev-ref HEAD

git add -A

$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# always commit (even if empty changes would fail otherwise)
git commit --allow-empty -m "mirror sync $timestamp"

# HARD overwrite remote branch
git push --force origin $branch

Write-Host "REMOTE NOW MATCHES LOCAL (forced) at $timestamp" -ForegroundColor Green