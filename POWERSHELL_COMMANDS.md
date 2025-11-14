# PowerShell Command Reference

Since you're on Windows using PowerShell, here are the correct commands for common tasks:

## Git Commands

### Check what will be committed
```powershell
git status
git diff --cached
```

### Check if config.py is accidentally staged
```powershell
# Method 1: Using Select-String (PowerShell native)
git diff --cached --name-only | Select-String config.py

# Method 2: Using findstr (Windows native)
git diff --cached --name-only | findstr config.py

# Method 3: Using Where-Object (PowerShell)
git diff --cached --name-only | Where-Object { $_ -like "*config.py*" }
```

### See what files are ignored
```powershell
git status --ignored
```

### Check if a file is ignored
```powershell
git check-ignore -v *.csv
git check-ignore -v config.py
```

## Python Commands

### Run test script
```powershell
python test_refactored_structure.py
```

### Test a specific import
```powershell
python -c "from qb_research.utils.data_loading import load_csv_safe; print('OK')"
```

### Check Python path
```powershell
python -c "import sys; print(sys.path)"
```

## File Operations

### Search for text in files (PowerShell)
```powershell
# Search in all Python files
Get-ChildItem -Recurse -Filter *.py | Select-String "pattern"

# Search in specific file
Select-String -Path "file.py" -Pattern "pattern"
```

### List files matching pattern
```powershell
# Find all test files
Get-ChildItem -Filter test_*.py

# Find all CSV files (to verify they're ignored)
Get-ChildItem -Filter *.csv
```

## Common Workflows

### Before committing - full checklist
```powershell
# 1. Check status
git status

# 2. Review staged changes
git diff --cached

# 3. Verify config.py is NOT staged
$staged = git diff --cached --name-only
if ($staged -contains "config.py") {
    Write-Host "WARNING: config.py is staged! Unstage it with: git reset HEAD config.py" -ForegroundColor Red
} else {
    Write-Host "OK: config.py is not staged" -ForegroundColor Green
}

# 4. Verify CSV files are ignored
$csvFiles = git status --ignored | Select-String "\.csv"
if ($csvFiles) {
    Write-Host "OK: CSV files are being ignored" -ForegroundColor Green
}
```

### Safe commit workflow
```powershell
# Stage only code and docs
git add qb_research/
git add QB_research.py
git add *.md
git add .gitignore
git add test_*.py
git add config.example.py

# Verify what's staged
git status

# Commit
git commit -m "Your commit message"

# Push
git push origin main
```

## Differences from Bash/Linux

| Bash/Linux | PowerShell |
|------------|------------|
| `grep pattern` | `Select-String pattern` or `findstr pattern` |
| `grep -r pattern .` | `Get-ChildItem -Recurse \| Select-String pattern` |
| `ls *.py` | `Get-ChildItem *.py` or `dir *.py` |
| `cat file` | `Get-Content file` or `type file` |
| `\|\|` (OR) | `-or` |
| `&&` (AND) | `-and` or `;` |

## Quick Reference

### Check if file exists
```powershell
Test-Path "config.py"
```

### Copy file
```powershell
Copy-Item config.example.py config.py
```

### View file content
```powershell
Get-Content file.py
# Or with pager:
Get-Content file.py | More
```

### Count lines in file
```powershell
(Get-Content file.py).Count
```

