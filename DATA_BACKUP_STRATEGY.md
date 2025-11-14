# Data Backup Strategy for QB Research

## Overview

Your CSV data files are **valuable research assets** that should be backed up redundantly, but **NOT in GitHub**. This guide explains how to protect your data.

## Why NOT GitHub?

- **GitHub has file size limits** (100MB warning, 50MB hard limit)
- **Git history gets huge** with large CSV files
- **Slows down git operations** (clone, pull, push)
- **Data changes frequently** - not ideal for version control
- **Security** - if repo is public, data could be exposed

## ✅ Recommended Backup Strategy

### Option 1: Cloud Storage (Recommended)

**Best for:** Automatic sync, redundancy, easy access

#### Google Drive / OneDrive / Dropbox
1. Create a folder: `QB_Research_Data_Backup`
2. Copy your data directories:
   ```
   QB_Research_Data_Backup/
   ├── QB_Data/              # Individual QB files
   ├── all_seasons_df.csv
   ├── qb_seasons_payment_labeled.csv
   ├── qb_seasons_payment_labeled_era_adjusted.csv
   ├── first_round_qbs_with_picks.csv
   ├── player_ids.csv
   ├── cache/                # Contract mappings, etc.
   └── [other important CSVs]
   ```
3. Enable automatic sync
4. **Advantage:** Automatic backup, accessible from anywhere, version history

#### AWS S3 / Google Cloud Storage
- **Best for:** Large datasets, automated backups
- Set up lifecycle policies for automatic backups
- More technical setup required

### Option 2: External Hard Drive / USB

**Best for:** Local backup, fast access, one-time cost

1. Create folder structure on external drive
2. Copy data directories
3. **Schedule regular backups** (weekly/monthly)
4. **Advantage:** Fast, no internet required, physical control
5. **Disadvantage:** Manual process, can be lost/damaged

### Option 3: Multiple Locations (3-2-1 Rule)

**Best practice:** 3 copies, 2 different media, 1 offsite

1. **Primary:** Your working directory (local)
2. **Backup 1:** Cloud storage (Google Drive/OneDrive)
3. **Backup 2:** External drive (updated monthly)
4. **Backup 3:** Another cloud service (optional)

## 📁 What to Backup

### Critical Data (Must Backup)
```
QB_Data/                          # Individual QB season files
all_seasons_df.csv                # Master dataset
qb_seasons_payment_labeled.csv   # Payment-labeled data
qb_seasons_payment_labeled_era_adjusted.csv  # Era-adjusted data
first_round_qbs_with_picks.csv    # Draft data
player_ids.csv                    # Player ID mapping
QB_contract_data.csv              # Contract information
cache/
  ├── contract_player_id_mapping.csv
  └── contract_mapping_report.txt
```

### Important Outputs (Consider Backing Up)
```
comp_analysis_output/             # Comparison analyses
KNN_surfaces/                     # KNN surface files
year_weights_ridge_results/       # Regression results
wins_ridge_results/               # Wins prediction results
```

### Less Critical (Can Regenerate)
- Test output files
- Temporary analysis files
- Scatter plot images

## 🔧 Automated Backup Script

Create a PowerShell script to automate backups:

```powershell
# backup_data.ps1
# Run this script to backup your data to cloud storage

$sourceDir = "C:\Users\Andre\.cursor\worktrees\football\63gZL"
$backupDir = "$env:USERPROFILE\OneDrive\QB_Research_Data_Backup"  # Adjust path

# Create backup directory if it doesn't exist
if (-not (Test-Path $backupDir)) {
    New-Item -ItemType Directory -Path $backupDir -Force
}

# Directories to backup
$dirsToBackup = @(
    "QB_Data",
    "cache",
    "comp_analysis_output"
)

# Files to backup
$filesToBackup = @(
    "all_seasons_df.csv",
    "qb_seasons_payment_labeled.csv",
    "qb_seasons_payment_labeled_era_adjusted.csv",
    "first_round_qbs_with_picks.csv",
    "first_round_qbs.csv",
    "player_ids.csv",
    "QB_contract_data.csv",
    "season_records.csv",
    "season_averages.csv",
    "best_seasons_df.csv",
    "salary_cap_by_year.csv",
    "era_adjustment_factors.csv"
)

Write-Host "Starting backup..." -ForegroundColor Green
Write-Host "Source: $sourceDir" -ForegroundColor Cyan
Write-Host "Destination: $backupDir" -ForegroundColor Cyan

# Backup directories
foreach ($dir in $dirsToBackup) {
    $sourcePath = Join-Path $sourceDir $dir
    $destPath = Join-Path $backupDir $dir
    
    if (Test-Path $sourcePath) {
        Write-Host "Backing up directory: $dir" -ForegroundColor Yellow
        Copy-Item -Path $sourcePath -Destination $destPath -Recurse -Force
        Write-Host "  ✓ $dir backed up" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $dir not found, skipping" -ForegroundColor Yellow
    }
}

# Backup files
foreach ($file in $filesToBackup) {
    $sourcePath = Join-Path $sourceDir $file
    $destPath = Join-Path $backupDir $file
    
    if (Test-Path $sourcePath) {
        Write-Host "Backing up file: $file" -ForegroundColor Yellow
        Copy-Item -Path $sourcePath -Destination $destPath -Force
        Write-Host "  ✓ $file backed up" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $file not found, skipping" -ForegroundColor Yellow
    }
}

Write-Host "`nBackup complete!" -ForegroundColor Green
Write-Host "Backup location: $backupDir" -ForegroundColor Cyan
```

## 📅 Backup Schedule Recommendations

### Daily (Automatic)
- Use cloud storage sync (OneDrive/Google Drive) for automatic daily sync

### Weekly (Manual or Scheduled Task)
- Run backup script to external drive
- Verify backup integrity

### Monthly
- Full backup to external drive
- Verify all critical files are backed up
- Test restore process (pick one file, restore it)

## 🚨 Disaster Recovery Plan

### If You Lose Your Data:

1. **Check cloud backup first** (OneDrive/Google Drive)
2. **Check external drive backup**
3. **Check if data is in git history** (unlikely, but check)
4. **Recreate from source** (if you have web scraping scripts)

### Prevention:
- **Never delete** original data files
- **Keep backups in multiple locations**
- **Test restore process** periodically
- **Document data sources** (where you got the data from)

## 📝 Data Inventory

Create a file listing what data you have and where it came from:

```markdown
# Data Inventory

## Critical Data Files

### all_seasons_df.csv
- **Source:** Created from QB_Data/*.csv files
- **Last Updated:** [Date]
- **Size:** [Size]
- **Backup Locations:** OneDrive, External Drive

### QB_Data/
- **Source:** Scraped from Pro Football Reference
- **Last Updated:** [Date]
- **File Count:** [Number]
- **Backup Locations:** OneDrive, External Drive

[Continue for each important file...]
```

## 🔐 Security Considerations

- **Don't commit sensitive data** (player contracts with personal info)
- **Encrypt backups** if containing sensitive information
- **Use strong passwords** for cloud storage accounts
- **Enable 2FA** on cloud storage accounts

## ✅ Quick Checklist

- [ ] Identify all critical data files
- [ ] Set up cloud storage backup (OneDrive/Google Drive)
- [ ] Create backup script
- [ ] Test backup process
- [ ] Schedule regular backups
- [ ] Create data inventory document
- [ ] Test restore process
- [ ] Document data sources

## 💡 Pro Tips

1. **Use symbolic links** if you want data in a synced folder but code elsewhere:
   ```powershell
   # Create symlink from project to OneDrive
   New-Item -ItemType SymbolicLink -Path ".\data_backup" -Target "$env:USERPROFILE\OneDrive\QB_Research_Data_Backup"
   ```

2. **Version your backups** with dates:
   ```
   QB_Research_Data_Backup_2024-01-15/
   QB_Research_Data_Backup_2024-02-15/
   ```

3. **Compress old backups** to save space:
   ```powershell
   Compress-Archive -Path "QB_Research_Data_Backup" -DestinationPath "backup_2024-01-15.zip"
   ```

4. **Use Git LFS for large files** (if you MUST version control some data):
   - Only for small, critical reference files
   - Not recommended for all CSV files

