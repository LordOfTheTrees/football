# Quick backup runner - uses your Proton Drive location
# Just double-click this file or run: .\run_backup.ps1

$backupLocation = "D:\proton drive\My files\Backup Docs\Personal Coding\QB research backup"

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "QB Research Data Backup" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host ""
Write-Host "Backup Location: $backupLocation" -ForegroundColor White
Write-Host ""

# Run the backup script with your location
& ".\backup_data.ps1" -BackupLocation $backupLocation

Write-Host ""
Write-Host "Backup complete! Your data is now in Proton Drive and will sync to cloud." -ForegroundColor Green
Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

