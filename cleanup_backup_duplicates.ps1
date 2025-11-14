# cleanup_backup_duplicates.ps1
# Script to clean up duplicate comp_analysis_output folders in backup

$backupPath = "D:\proton drive\My files\Backup Docs\Personal Coding\QB research backup"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Cleaning Up Backup Duplicates" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""

# Find all instances of comp_analysis_output in backup
$allInstances = Get-ChildItem -Path $backupPath -Recurse -Directory -Filter "comp_analysis_output" -ErrorAction SilentlyContinue

if ($allInstances.Count -eq 0) {
    Write-Host "No comp_analysis_output folders found in backup." -ForegroundColor Green
    exit 0
}

Write-Host "Found $($allInstances.Count) instance(s) of comp_analysis_output:" -ForegroundColor Yellow
foreach ($instance in $allInstances) {
    Write-Host "  - $($instance.FullName)" -ForegroundColor White
}

Write-Host ""
$rootInstance = Join-Path $backupPath "comp_analysis_output"

# Keep only the root-level instance
$toRemove = $allInstances | Where-Object { $_.FullName -ne $rootInstance }

if ($toRemove.Count -eq 0) {
    Write-Host "No duplicates found - all instances are in expected locations." -ForegroundColor Green
    exit 0
}

Write-Host "The following duplicate(s) will be removed:" -ForegroundColor Yellow
foreach ($duplicate in $toRemove) {
    Write-Host "  - $($duplicate.FullName)" -ForegroundColor Red
}

Write-Host ""
$confirm = Read-Host "Remove these duplicates? (yes/no)"

if ($confirm -eq "yes") {
    foreach ($duplicate in $toRemove) {
        Write-Host "Removing: $($duplicate.FullName)" -ForegroundColor Yellow
        Remove-Item -Path $duplicate.FullName -Recurse -Force -ErrorAction Stop
        Write-Host "  OK: Removed" -ForegroundColor Green
    }
    Write-Host "`nCleanup complete!" -ForegroundColor Green
} else {
    Write-Host "Cleanup cancelled." -ForegroundColor Yellow
}

