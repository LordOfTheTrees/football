# Development Guide: QB Research Package

## Overview

This guide explains how to work with the refactored QB research codebase going forward.

## Package Structure

```
qb_research/
├── utils/              # Core utilities (data loading, name matching, caching, debug, exploratory)
├── data/               # Data loading, mapping, building, validation
├── preprocessing/      # Feature engineering and data preparation
├── analysis/           # Statistical analysis and era adjustment
├── modeling/           # Prediction models and surface models
├── comparisons/        # QB comparison and trajectory matching
└── exports/            # Tableau and data export functions
```

## Import Patterns

### ✅ Recommended: Use New Package Structure

```python
# Good - use new structure
from qb_research.utils.data_loading import load_csv_safe
from qb_research.data.loaders import load_contract_data
from qb_research.preprocessing.feature_engineering import prepare_qb_payment_data
from qb_research.modeling.prediction_models import ridge_regression_payment_prediction
```

### ⚠️ Still Works: Backward Compatibility

```python
# Still works - backward compatible
from QB_research import load_csv_safe, prepare_qb_payment_data
```

**Note:** Backward compatibility is maintained but deprecated. Migrate to new imports when convenient.

## Development Workflow

### 1. Making Changes

**When adding new functions:**
- Place them in the appropriate module based on functionality
- Update the module's `__init__.py` to export the function
- Add backward compatibility import in `QB_research.py` (optional, for transition period)

**Example:**
```python
# qb_research/utils/new_module.py
def new_utility_function():
    """New utility function."""
    pass

# qb_research/utils/__init__.py
from .new_module import new_utility_function
```

### 2. Testing Your Changes

```bash
# Run comprehensive test suite
python test_refactored_structure.py

# Test specific module
python -c "from qb_research.your.module import your_function; print('OK')"
```

### 3. Git Workflow

#### ✅ What to Commit

**Code and Structure:**
- All Python files in `qb_research/`
- `QB_research.py` (backward compatibility layer)
- Standalone scripts: `qb_comp_tool.py`, `standings_scraper.py`, etc.
- Documentation: `*.md` files
- Configuration examples: `config.example.py`
- Requirements: `requirements.txt` (if you have one)

**Example commit:**
```bash
git add qb_research/
git add QB_research.py
git add *.md
git add config.example.py
git commit -m "Refactor: Organize code into qb_research package structure"
```

#### ❌ What NOT to Commit

**Data Files:**
- `*.csv` files (data, outputs, results)
- `QB_Data/` directory
- `cache/` directory
- Output directories: `comp_analysis_output/`, `KNN_surfaces/`, etc.

**Sensitive Files:**
- `config.py` (contains API keys)
- Any file with credentials

**Generated Files:**
- `__pycache__/` directories
- `*.pyc` files
- Test output files

**Large Files:**
- `*.twbx`, `*.twb` (Tableau workbooks)

### 4. Before Committing

**Checklist:**
- [ ] Run `python test_refactored_structure.py` - all tests pass
- [ ] Verify no sensitive data in commits: `git diff` before committing
- [ ] Check `.gitignore` is up to date
- [ ] Ensure `config.py` is not tracked (should be in `.gitignore`)

**Verify what will be committed:**
```powershell
# PowerShell commands
git status
git diff --cached  # Review staged changes

# Check if config.py is accidentally staged
git diff --cached --name-only | Select-String config.py
# Should return nothing - if it shows config.py, unstage it with:
# git reset HEAD config.py
```

### 5. Branching Strategy (Optional)

For larger features, consider branching:
```bash
# Create feature branch
git checkout -b feature/new-analysis-module

# Make changes, test, commit
# ...

# Merge back to main
git checkout main
git merge feature/new-analysis-module
```

## Running the Pipeline

The refactored code maintains backward compatibility, so existing scripts should work:

```python
# Old way (still works)
from QB_research import prepare_qb_payment_data
prepare_qb_payment_data()

# New way (recommended)
from qb_research.preprocessing.feature_engineering import prepare_qb_payment_data
prepare_qb_payment_data()
```

## Common Tasks

### Adding a New Analysis Function

1. Create or update module in `qb_research/analysis/`
2. Add function to module
3. Export in `qb_research/analysis/__init__.py`
4. Test: `python -c "from qb_research.analysis import your_function"`

### Adding a New Data Loader

1. Add to `qb_research/data/loaders.py`
2. Export in `qb_research/data/__init__.py` (if needed)
3. Test with actual data file

### Debugging Import Issues

```python
# Check if module exists
import qb_research.utils.data_loading
print(dir(qb_research.utils.data_loading))

# Check if function exists
from qb_research.utils.data_loading import load_csv_safe
print(load_csv_safe)
```

## Migration Path

**Phase 1 (Current):** Backward compatibility maintained
- Old imports still work
- New imports available
- Both can coexist

**Phase 2 (Future):** Gradual migration
- Update scripts to use new imports
- Test thoroughly
- Remove backward compatibility layer when ready

## Troubleshooting

### ImportError: No module named 'config'

**Solution:** Create `config.py` from `config.example.py`:
```bash
cp config.example.py config.py
# Edit config.py with your API keys
```

### ImportError: cannot import name 'X' from 'qb_research.Y'

**Solution:** 
1. Check if function exists in the module file
2. Check if it's exported in `__init__.py`
3. Verify spelling and module path

### PowerShell: grep command not found

**Solution:** In PowerShell, use `Select-String` instead of `grep`:
```powershell
# Instead of: git diff | grep config.py
git diff | Select-String config.py

# Or use findstr (Windows native):
git diff | findstr config.py
```

### Functions work but tests fail

**Solution:** 
- Check if `config.py` exists (some tests skip if missing)
- Verify you're in the project root directory
- Check Python path: `python -c "import sys; print(sys.path)"`

## Best Practices

1. **Use new imports** in new code
2. **Test imports** before committing
3. **Keep functions focused** - one responsibility per function
4. **Document functions** with docstrings
5. **Update tests** when adding new functionality
6. **Review git status** before committing to avoid data files

## Questions?

- Check `REFACTORING_PLAN.md` for structure details
- Check `HOW_TO_RUN_PIPELINE.md` for pipeline usage
- Review module `__init__.py` files for available exports

