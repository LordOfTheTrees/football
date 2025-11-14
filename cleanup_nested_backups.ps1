# cleanup_nested_backups.ps1
# Script to clean up nested directory structures created by previous backup runs

$backupPath = "D:\proton drive\My files\Backup Docs\Personal Coding\QB research backup"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Cleaning Up Nested Backup Directories" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Find nested QB_Data directories
$nestedQBData = Get-ChildItem -Path $backupPath -Recurse -Directory -Filter "QB_Data" -ErrorAction SilentlyContinue | 
    Where-Object { $_.FullName -ne (Join-Path $backupPath "QB_Data") }

# Find nested comp_analysis_output directories
$nestedCompAnalysis = Get-ChildItem -Path $backupPath -Recurse -Directory -Filter "comp_analysis_output" -ErrorAction SilentlyContinue | 
    Where-Object { $_.FullName -ne (Join-Path $backupPath "comp_analysis_output") }

$toRemove = @()
if ($nestedQBData) {
    $toRemove += $nestedQBData
}
if ($nestedCompAnalysis) {
    $toRemove += $nestedCompAnalysis
}

if ($toRemove.Count -eq 0) {
    Write-Host "No nested directories found. Backup structure is clean." -ForegroundColor Green
    exit 0
}

Write-Host "Found $($toRemove.Count) nested directory(ies) to remove:" -ForegroundColor Yellow
foreach ($item in $toRemove) {
    Write-Host "  - $($item.FullName)" -ForegroundColor Red
}

Write-Host ""
$confirm = Read-Host "Remove these nested directories? (yes/no)"

if ($confirm -eq "yes") {
    foreach ($item in $toRemove) {
        Write-Host "Removing: $($item.FullName)" -ForegroundColor Yellow
        try {
            Remove-Item -Path $item.FullName -Recurse -Force -ErrorAction Stop
            Write-Host "  OK: Removed" -ForegroundColor Green
        } catch {
            Write-Host "  ERROR: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
    Write-Host "`nCleanup complete!" -ForegroundColor Green
} else {
    Write-Host "Cleanup cancelled." -ForegroundColor Yellow
}

