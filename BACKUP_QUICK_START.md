# Quick Start: Backup Your Data NOW

## Your Current Data

You have:
- **19 CSV files** in your project root
- **73 files** in your `QB_Data/` directory
- Plus cache files, analysis outputs, etc.

**This is valuable research data - let's protect it!**

## 🚀 Quick Setup (Already Configured!)

Your backup location is already set up:
- **Location:** `D:\proton drive\My files\Backup Docs\Personal Coding\QB research backup`
- **Status:** ✅ Folder exists and is ready
- **Sync:** Proton Drive will automatically sync to cloud

### Step 1: Test the Backup Script (Dry Run)

```powershell
# First, do a dry run to see what will be backed up (no files copied)
.\backup_data.ps1 -DryRun

# If it looks good, run the actual backup
.\backup_data.ps1

# Or use the quick runner (double-click run_backup.ps1)
.\run_backup.ps1
```

### Step 2: Verify the Backup

```powershell
$backupPath = "D:\proton drive\My files\Backup Docs\Personal Coding\QB research backup"

# Check that files were copied
Get-ChildItem $backupPath -Recurse | Measure-Object

# Compare file counts
Write-Host "Source CSV files: $((Get-ChildItem -Filter *.csv).Count)"
Write-Host "Backup CSV files: $((Get-ChildItem -Path '$backupPath' -Filter *.csv -Recurse).Count)"
```

## 📅 Set Up Automatic Backups

**Good News:** Your Proton Drive folder already syncs automatically! Every time you run the backup script, Proton Drive will sync the files to the cloud.

### Option 1: Scheduled Automatic Backups (Recommended)

1. **Move your data to a synced folder:**
   ```powershell
   # Create folder in OneDrive
   $dataFolder = "$env:USERPROFILE\OneDrive\QB_Research_Data"
   New-Item -ItemType Directory -Path $dataFolder -Force
   
   # Copy data there
   Copy-Item -Path ".\QB_Data" -Destination "$dataFolder\QB_Data" -Recurse
   Copy-Item -Path ".\*.csv" -Destination $dataFolder
   ```

2. **Use symbolic links** to keep data accessible from your project:
   ```powershell
   # Backup original
   Move-Item -Path ".\QB_Data" -Destination ".\QB_Data.backup"
   
   # Create symlink from project to OneDrive
   New-Item -ItemType SymbolicLink -Path ".\QB_Data" -Target "$env:USERPROFILE\OneDrive\QB_Research_Data\QB_Data"
   ```

3. **OneDrive will automatically sync** - your data is now backed up!

### Option 2: Scheduled Task (Weekly Backup)

```powershell
# Create a scheduled task to run backup weekly
$action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File `"$PWD\backup_data.ps1`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 2am
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive
Register-ScheduledTask -TaskName "QB Research Data Backup" -Action $action -Trigger $trigger -Principal $principal -Description "Weekly backup of QB research data"
```

## ✅ Recommended Setup

**Best Practice: Multiple Backups**

1. **Primary Backup:** Proton Drive (automatic, daily) ✅ **YOU HAVE THIS!**
   - Your backup folder: `D:\proton drive\My files\Backup Docs\Personal Coding\QB research backup`
   - Proton Drive automatically syncs to cloud
   - Data is always backed up

2. **Secondary Backup:** External Drive (monthly, optional)
   - Run `backup_data.ps1` with different location monthly
   - Store drive in safe location

3. **Tertiary Backup:** Another Cloud Service (optional)
   - OneDrive, Google Drive, AWS S3, etc.
   - For extra redundancy

## 🔍 What Gets Backed Up?

The script backs up:

**Critical Data:**
- `QB_Data/` - All individual QB files (73 files)
- `cache/` - Contract mappings
- All CSV files in root (19 files)
- Era adjustment factors
- Year weights

**Output Directories:**
- `comp_analysis_output/` - Comparison analyses
- `KNN_surfaces/` - KNN surface files
- `article_tableaus/` - Tableau workbooks for articles
- `scatter_plots/` - Scatter plot visualizations

**Regression Results (Consolidated):**
All regression folders are backed up into a single `regression_results/` folder:
- `wins_prediction_min_variables/`
- `payment_prediction_min_variables/`
- `wins_ridge_results/`
- `year_weights_ridge_results/`
- `regression_variable_statistics/`
- `team+personal_payment_prediction_logistic_results/`
- `complete_variables_wins_payment/`

## ⚠️ Important Notes

1. **Don't put data in Git** - We've already set up `.gitignore` for this
2. **Test your restore** - Periodically verify you can restore files
3. **Keep backups current** - Update monthly at minimum
4. **Document data sources** - Know where your data came from

## 🆘 If You Lose Data

1. Check OneDrive/Google Drive first
2. Check external drive backup
3. Check if you can regenerate from source (web scraping)
4. Check git history (unlikely, but possible for small files)

## 📝 Next Steps

1. ✅ **Run `.\backup_data.ps1 -DryRun`** to see what will be backed up
2. ✅ **Run `.\backup_data.ps1`** to create your first backup (or use `.\run_backup.ps1`)
3. ✅ **Verify backup** - Check that files are in your Proton Drive folder
4. ✅ **Set up scheduled task** (optional) - For weekly automatic backups
5. ✅ **Test restore** - Pick one file, verify you can restore it from backup

## 💡 Pro Tip

Create a data inventory document:
```markdown
# Data Inventory - Last Updated: [Date]

## Critical Files
- all_seasons_df.csv - Created: [Date], Size: [Size], Source: QB_Data aggregation
- QB_Data/ - 73 files, Last updated: [Date], Source: Pro Football Reference scraping

[Document each important file]
```

This helps you know what to restore if disaster strikes!

