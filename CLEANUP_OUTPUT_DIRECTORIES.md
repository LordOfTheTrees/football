# Removing Output Directories from Git (While Keeping Locally)

## Current Status

✅ **Good news**: All output directories are already properly ignored by `.gitignore` and are **not currently tracked** by git.

The following directories exist locally but are not in the repository:
- `payment_heat_map_probabilities/`
- `KNN_surfaces/`
- `comp_analysis_output/`
- `wins_prediction_min_variables/`
- `regression_variable_statistics/`
- `year_weights_ridge_results/`
- `wins_ridge_results/`
- `team+personal_payment_prediction_logistic_results/`
- `complete_variables_wins_payment/`
- `scatter_plots/`

## If You Need to Remove Previously Tracked Directories

If these directories were previously committed to git (before being added to `.gitignore`), use these commands to remove them from git tracking while keeping them locally:

### PowerShell Commands

```powershell
# Remove specific directory from git tracking (keeps files locally)
git rm -r --cached payment_heat_map_probabilities
git rm -r --cached KNN_surfaces
git rm -r --cached comp_analysis_output
git rm -r --cached wins_prediction_min_variables
git rm -r --cached regression_variable_statistics
git rm -r --cached year_weights_ridge_results
git rm -r --cached wins_ridge_results
git rm -r --cached "team+personal_payment_prediction_logistic_results"
git rm -r --cached complete_variables_wins_payment
git rm -r --cached scatter_plots

# Or remove all at once
git rm -r --cached payment_heat_map_probabilities KNN_surfaces comp_analysis_output wins_prediction_min_variables regression_variable_statistics year_weights_ridge_results wins_ridge_results "team+personal_payment_prediction_logistic_results" complete_variables_wins_payment scatter_plots
```

### What This Does

- `git rm -r --cached` removes files/directories from git's index (staging area)
- The `--cached` flag means it **doesn't delete the files from your local disk**
- After running this, the directories will remain on your computer but won't be tracked by git
- Future commits will remove them from the repository

### Verify They're Ignored

After removing from tracking, verify they're properly ignored:

```powershell
# Check if directories are ignored
git check-ignore -v payment_heat_map_probabilities
git check-ignore -v KNN_surfaces
# etc.

# Verify they're not tracked
git ls-files | Select-String "payment_heat_map_probabilities"
# Should return nothing
```

### Commit the Changes

After removing from tracking, commit the changes:

```powershell
git commit -m "Remove output directories from git tracking (keep locally)"
git push
```

## Important Notes

1. **Local files are preserved**: The `--cached` flag ensures your local files stay intact
2. **`.gitignore` is already set up**: These directories are already in `.gitignore`, so they won't be accidentally added back
3. **Others will need to pull**: When others pull your changes, the directories will be removed from their local repos too (but they can recreate them by running your scripts)
4. **Your scripts still work**: Since the directories exist locally, all your scripts will continue to work normally

## Quick Check Script

Run this to see if any output directories are currently tracked:

```powershell
$dirs = @('payment_heat_map_probabilities', 'KNN_surfaces', 'comp_analysis_output', 'wins_prediction_min_variables', 'regression_variable_statistics', 'year_weights_ridge_results', 'wins_ridge_results', 'team+personal_payment_prediction_logistic_results', 'complete_variables_wins_payment', 'scatter_plots')

foreach ($dir in $dirs) {
    $tracked = git ls-files $dir
    if ($tracked) {
        Write-Host "⚠️  $dir is tracked:" -ForegroundColor Yellow
        $tracked | ForEach-Object { Write-Host "  $_" }
    } else {
        Write-Host "✅ $dir is not tracked (properly ignored)" -ForegroundColor Green
    }
}
```

