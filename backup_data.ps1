# backup_data.ps1
# Automated backup script for QB Research data
# Run this script to backup your data to cloud storage or external drive

param(
    [string]$BackupLocation = "D:\proton drive\My files\Backup Docs\Personal Coding\QB research backup",
    [switch]$DryRun = $false
)

# Get the script directory (assumes script is in project root)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceDir = $scriptDir

# Create backup directory if it doesn't exist
if (-not (Test-Path $BackupLocation)) {
    if (-not $DryRun) {
        Write-Host "Creating backup directory: $BackupLocation" -ForegroundColor Yellow
        New-Item -ItemType Directory -Path $BackupLocation -Force | Out-Null
    } else {
        Write-Host "[DRY RUN] Would create: $BackupLocation" -ForegroundColor Cyan
    }
}

Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "QB Research Data Backup" -ForegroundColor Green
Write-Host "=" * 80 -ForegroundColor Cyan
Write-Host "Source: $sourceDir" -ForegroundColor White
Write-Host "Destination: $BackupLocation" -ForegroundColor White
if ($DryRun) {
    Write-Host "MODE: DRY RUN (no files will be copied)" -ForegroundColor Yellow
}
Write-Host ""

# Directories to backup
$dirsToBackup = @(
    "QB_Data",
    "cache"
)

# Critical files to backup
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
    "era_adjustment_factors.csv",
    "year_weights_total_yards_adj.csv",
    "year_weights_Pass_ANY_A_adj.csv"
)

# Output directories to backup
$outputDirsToBackup = @(
    "comp_analysis_output",
    "KNN_surfaces",
    "article_tableaus",
    "scatter_plots"
)

# Regression result directories - will be consolidated into "regression_results" folder
$regressionDirs = @(
    "wins_prediction_min_variables",
    "payment_prediction_min_variables",
    "wins_ridge_results",
    "year_weights_ridge_results",
    "regression_variable_statistics",
    "team+personal_payment_prediction_logistic_results",
    "complete_variables_wins_payment"
)

# Directories to exclude from data directories (QB_Data, cache) to avoid duplicates
$excludeFromDataDirs = @("comp_analysis_output", "KNN_surfaces", "article_tableaus", "scatter_plots") + $regressionDirs

$backedUpCount = 0
$skippedCount = 0
$errorCount = 0

# Helper function to copy directory with exclusions using robocopy (more reliable)
function Copy-DirectoryWithExclusions {
    param(
        [string]$SourcePath,
        [string]$DestPath,
        [string[]]$ExcludeDirs
    )
    
    # Ensure destination parent exists
    $destParent = Split-Path -Parent $DestPath
    if ($destParent -and -not (Test-Path $destParent)) {
        New-Item -ItemType Directory -Path $destParent -Force | Out-Null
    }
    
    # Remove destination if it exists to avoid nested structures
    if (Test-Path $DestPath) {
        Remove-Item -Path $DestPath -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    # Create fresh destination directory
    New-Item -ItemType Directory -Path $DestPath -Force | Out-Null
    
    if ($ExcludeDirs.Count -eq 0) {
        # No exclusions, use simple copy
        # Copy contents, not the directory itself
        Get-ChildItem -Path $SourcePath | Copy-Item -Destination $DestPath -Recurse -Force -ErrorAction Stop
        return
    }
    
    # Build robocopy exclude switches
    $excludeSwitches = @()
    foreach ($excludeDir in $ExcludeDirs) {
        $excludeSwitches += "/XD"
        $excludeSwitches += $excludeDir
    }
    
    # Use robocopy with exclusions
    # /E = copy subdirectories including empty ones
    # /IS = include same files (overwrite)
    # /IT = include tweaked files
    # /NFL = don't log file names
    # /NDL = don't log directory names
    # /NP = don't show progress percentage
    # /NJH = no job header
    # /NJS = no job summary
    # /R:0 = retry 0 times (fail fast)
    # /W:0 = wait 0 seconds between retries
    # Note: robocopy copies CONTENTS of source to destination when destination doesn't exist
    # Don't use trailing backslashes - robocopy handles this better without them
    # Ensure destination parent exists but destination itself should not exist (we removed it above)
    $robocopyArgs = @($SourcePath, $DestPath) + $excludeSwitches + @("/E", "/IS", "/IT", "/NFL", "/NDL", "/NP", "/NJH", "/NJS", "/R:0", "/W:0")
    
    # Capture robocopy output for debugging
    $robocopyOutput = & robocopy @robocopyArgs 2>&1
    $exitCode = $LASTEXITCODE
    
    # Robocopy exit codes: 0-7 are success, 8+ are errors
    # 0 = no files copied (source and dest are identical)
    # 1 = files copied successfully
    # 2-7 = additional files copied or extra files in destination
    # 8+ = errors
    if ($exitCode -ge 8) {
        # Try fallback method if robocopy fails
        Write-Host "    WARNING: Robocopy failed (exit code $exitCode), trying fallback method..." -ForegroundColor Yellow
        # Remove destination and try with Copy-Item
        if (Test-Path $DestPath) {
            Remove-Item -Path $DestPath -Recurse -Force -ErrorAction SilentlyContinue
        }
        New-Item -ItemType Directory -Path $DestPath -Force | Out-Null
        
        # Use Copy-Item with manual exclusion (recursive)
        # Filter out excluded directories at top level
        Get-ChildItem -Path $SourcePath | Where-Object {
            if ($_.PSIsContainer) {
                -not ($ExcludeDirs -contains $_.Name)
            } else {
                $true
            }
        } | ForEach-Object {
            $destItem = Join-Path $DestPath $_.Name
            if ($_.PSIsContainer) {
                # Recursively copy directory, checking exclusions at each level
                Copy-Item -Path $_.FullName -Destination $destItem -Recurse -Force -ErrorAction Stop
                # Remove any excluded subdirectories that might have been copied
                foreach ($excludeDir in $ExcludeDirs) {
                    $excludePath = Join-Path $destItem $excludeDir
                    if (Test-Path $excludePath) {
                        Remove-Item -Path $excludePath -Recurse -Force -ErrorAction SilentlyContinue
                    }
                }
            } else {
                Copy-Item -Path $_.FullName -Destination $destItem -Force -ErrorAction Stop
            }
        }
    }
}

# Function to backup item
function Backup-Item {
    param(
        [string]$SourcePath,
        [string]$DestPath,
        [string]$ItemName,
        [bool]$IsDirectory = $false,
        [string[]]$ExcludeSubdirs = @()
    )
    
    if (-not (Test-Path $SourcePath)) {
        Write-Host "  SKIP: $ItemName (not found)" -ForegroundColor Yellow
        $script:skippedCount++
        return
    }
    
    try {
        if ($DryRun) {
            Write-Host "  [DRY RUN] Would backup: $ItemName" -ForegroundColor Cyan
            if ($IsDirectory -and $ExcludeSubdirs.Count -gt 0) {
                Write-Host "    (Would exclude: $($ExcludeSubdirs -join ', '))" -ForegroundColor Yellow
            }
            $script:backedUpCount++
        } else {
            if ($IsDirectory) {
                # Use robocopy for reliable exclusions
                Copy-DirectoryWithExclusions -SourcePath $SourcePath -DestPath $DestPath -ExcludeDirs $ExcludeSubdirs
            } else {
                $destDir = Split-Path -Parent $DestPath
                if (-not (Test-Path $destDir)) {
                    New-Item -ItemType Directory -Path $destDir -Force | Out-Null
                }
                Copy-Item -Path $SourcePath -Destination $DestPath -Force -ErrorAction Stop
            }
            Write-Host "  OK: $ItemName" -ForegroundColor Green
            $script:backedUpCount++
        }
    } catch {
        Write-Host "  ERROR: $ItemName - $($_.Exception.Message)" -ForegroundColor Red
        $script:errorCount++
    }
}

# Backup directories (QB_Data, cache) - exclude output and regression directories
Write-Host "Backing up directories..." -ForegroundColor Cyan
foreach ($dir in $dirsToBackup) {
    $sourcePath = Join-Path $sourceDir $dir
    $destPath = Join-Path $BackupLocation $dir
    
    # Check if this directory contains any directories we're backing up separately
    $excludeSubdirs = @()
    foreach ($excludeDir in $excludeFromDataDirs) {
        if (Test-Path (Join-Path $sourcePath $excludeDir)) {
            $excludeSubdirs += $excludeDir
            Write-Host "  NOTE: $dir contains $excludeDir - will exclude nested copy" -ForegroundColor Yellow
        }
    }
    
    Backup-Item -SourcePath $sourcePath -DestPath $destPath -ItemName $dir -IsDirectory $true -ExcludeSubdirs $excludeSubdirs
}

# Backup files
Write-Host "`nBacking up files..." -ForegroundColor Cyan
foreach ($file in $filesToBackup) {
    $sourcePath = Join-Path $sourceDir $file
    $destPath = Join-Path $BackupLocation $file
    Backup-Item -SourcePath $sourcePath -DestPath $destPath -ItemName $file -IsDirectory $false
}

# Backup output directories
if ($outputDirsToBackup.Count -gt 0) {
    Write-Host "`nBacking up output directories..." -ForegroundColor Cyan
    foreach ($dir in $outputDirsToBackup) {
        $sourcePath = Join-Path $sourceDir $dir
        $destPath = Join-Path $BackupLocation $dir
        
        # Check if this directory contains any other output directories (to exclude nested duplicates)
        $excludeSubdirs = @()
        foreach ($otherDir in $outputDirsToBackup) {
            if ($dir -ne $otherDir -and (Test-Path (Join-Path $sourcePath $otherDir))) {
                $excludeSubdirs += $otherDir
                Write-Host "  NOTE: $dir contains $otherDir - will exclude nested copy" -ForegroundColor Yellow
            }
        }
        # Also exclude regression directories
        foreach ($regDir in $regressionDirs) {
            if (Test-Path (Join-Path $sourcePath $regDir)) {
                $excludeSubdirs += $regDir
                Write-Host "  NOTE: $dir contains $regDir - will exclude nested copy" -ForegroundColor Yellow
            }
        }
        
        Backup-Item -SourcePath $sourcePath -DestPath $destPath -ItemName $dir -IsDirectory $true -ExcludeSubdirs $excludeSubdirs
    }
}

# Backup regression results (consolidated into single folder)
$excludeFromRegression = @("comp_analysis_output", "KNN_surfaces", "article_tableaus", "scatter_plots")

if ($regressionDirs.Count -gt 0) {
    Write-Host "`nBacking up regression results (consolidated)..." -ForegroundColor Cyan
    $regressionBackupPath = Join-Path $BackupLocation "regression_results"
    
    if (-not $DryRun) {
        if (-not (Test-Path $regressionBackupPath)) {
            New-Item -ItemType Directory -Path $regressionBackupPath -Force | Out-Null
        }
    } else {
        Write-Host "  [DRY RUN] Would create: regression_results\" -ForegroundColor Cyan
    }
    
    foreach ($dir in $regressionDirs) {
        $sourcePath = Join-Path $sourceDir $dir
        $destPath = Join-Path $regressionBackupPath $dir
        
        if (-not (Test-Path $sourcePath)) {
            Write-Host "  SKIP: $dir (not found)" -ForegroundColor Yellow
            $skippedCount++
            continue
        }
        
        try {
            if ($DryRun) {
                Write-Host "  [DRY RUN] Would backup: $dir -> regression_results\$dir" -ForegroundColor Cyan
                # Check for nested directories that should be excluded
                foreach ($excludeDir in $excludeFromRegression) {
                    if (Test-Path (Join-Path $sourcePath $excludeDir)) {
                        Write-Host "    (Would exclude nested: $excludeDir)" -ForegroundColor Yellow
                    }
                }
                $backedUpCount++
            } else {
                # Use Backup-Item function with exclusions to prevent nested duplicates
                Backup-Item -SourcePath $sourcePath -DestPath $destPath -ItemName "$dir -> regression_results\$dir" -IsDirectory $true -ExcludeSubdirs $excludeFromRegression
            }
        } catch {
            Write-Host "  ERROR: $dir - $($_.Exception.Message)" -ForegroundColor Red
            $errorCount++
        }
    }
}

# Summary
Write-Host "`n" + ("=" * 80) -ForegroundColor Cyan
Write-Host "Backup Summary" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "Backed up: $backedUpCount items" -ForegroundColor Green
Write-Host "Skipped: $skippedCount items" -ForegroundColor Yellow
Write-Host "Errors: $errorCount items" -ForegroundColor $(if ($errorCount -eq 0) { "Green" } else { "Red" })
Write-Host "`nBackup location: $BackupLocation" -ForegroundColor Cyan

if (-not $DryRun) {
    Write-Host "`nBackup complete!" -ForegroundColor Green
} else {
    Write-Host "`nDry run complete. Run without -DryRun to perform actual backup." -ForegroundColor Yellow
}
