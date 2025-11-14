# Backup Instructions - Quick Reference

## Your Backup Setup

**Backup Location:** `D:\proton drive\My files\Backup Docs\Personal Coding\QB research backup`

**Status:** ✅ Configured and ready to use

## How to Run Backup

### Method 1: Quick Run (Easiest)
```powershell
# Just double-click this file or run:
.\run_backup.ps1
```

### Method 2: Full Script
```powershell
# Test first (dry run - no files copied)
.\backup_data.ps1 -DryRun

# Run actual backup
.\backup_data.ps1
```

### Method 3: Custom Location
```powershell
# Backup to a different location
.\backup_data.ps1 -BackupLocation "E:\MyBackups\QB_Data"
```

## What Gets Backed Up?

✅ **Critical Data:**
- `QB_Data/` - All 73 individual QB files
- `cache/` - Contract mappings
- All CSV files in root (19 files)
- Era adjustment factors
- Year weights

✅ **Output Directories:**
- `comp_analysis_output/` - Comparison analyses
- `KNN_surfaces/` - KNN surface files
- `article_tableaus/` - Tableau workbooks for articles
- `scatter_plots/` - Scatter plot visualizations

✅ **Regression Results (Consolidated):**
All regression folders are backed up into a single `regression_results/` folder:
- `wins_prediction_min_variables/`
- `payment_prediction_min_variables/`
- `wins_ridge_results/`
- `year_weights_ridge_results/`
- `regression_variable_statistics/`
- `team+personal_payment_prediction_logistic_results/`
- `complete_variables_wins_payment/`

## Verify Backup

```powershell
$backupPath = "D:\proton drive\My files\Backup Docs\Personal Coding\QB research backup"

# Count files in backup
(Get-ChildItem $backupPath -Recurse -File).Count

# List what's backed up
Get-ChildItem $backupPath -Recurse | Select-Object FullName, Length, LastWriteTime
```

## Schedule Automatic Backups

### Weekly Backup (Recommended)

```powershell
# Create scheduled task for weekly backups (Sundays at 2am)
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File `"$PWD\backup_data.ps1`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 2am
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
Register-ScheduledTask -TaskName "QB Research Data Backup" -Action $action -Trigger $trigger -Principal $principal -Description "Weekly backup of QB research data to Proton Drive"
```

### Daily Backup (If Needed)

```powershell
# Change -Weekly to -Daily
$trigger = New-ScheduledTaskTrigger -Daily -At 2am
```

## Restore from Backup

```powershell
$backupPath = "D:\proton drive\My files\Backup Docs\Personal Coding\QB research backup"

# Restore a specific file
Copy-Item "$backupPath\all_seasons_df.csv" -Destination ".\all_seasons_df.csv" -Force

# Restore entire directory
Copy-Item "$backupPath\QB_Data" -Destination ".\QB_Data" -Recurse -Force
```

## Troubleshooting

### "Path not found" error
- Check that Proton Drive folder exists
- Verify the path: `D:\proton drive\My files\Backup Docs\Personal Coding\QB research backup`

### "Access denied" error
- Run PowerShell as Administrator
- Check Proton Drive sync status

### Files not syncing to cloud
- Check Proton Drive sync status
- Verify you're logged into Proton Drive
- Check internet connection

## Quick Commands

```powershell
# Run backup now
.\backup_data.ps1

# Test backup (dry run)
.\backup_data.ps1 -DryRun

# Check backup folder
Get-ChildItem "D:\proton drive\My files\Backup Docs\Personal Coding\QB research backup" -Recurse | Measure-Object

# View backup script help
Get-Help .\backup_data.ps1 -Full
```

