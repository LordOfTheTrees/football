# How to Run the Pipeline After Refactoring

## Current Way (Before Refactoring)

```bash
# Full rebuild
python QB_research.py

# Debug specific QB
python QB_research.py -q "Stafford"

# Debug specific stage
python QB_research.py -q "Stafford" --stage payment

# Skip rebuild, just debug
python QB_research.py -q "Stafford" --skip-rebuild
```

## After Refactoring - Multiple Options

### Option 1: Keep `QB_research.py` Working (Easiest Transition) ⭐ **RECOMMENDED**

**What happens**: `QB_research.py` becomes a thin wrapper that calls the new structure.

**How to run** (exactly the same as before):
```bash
# Full rebuild
python QB_research.py

# Debug specific QB
python QB_research.py -q "Stafford"

# All existing commands work the same
python QB_research.py -q "Stafford" --stage payment --skip-rebuild
```

**Implementation**: `QB_research.py` would look like:
```python
# QB_research.py (backward compatibility wrapper)
from qb_research.preprocessing.pipeline import main

if __name__ == "__main__":
    main()  # Just delegates to the new structure
```

**Pros**: 
- ✅ Zero learning curve
- ✅ All existing scripts/commands still work
- ✅ Can migrate gradually

**Cons**: 
- ⚠️ Still have the old file name (but it's just a thin wrapper)

---

### Option 2: Standalone Script at Root (Clean Long-Term)

**What happens**: Create a new `rebuild_pipeline.py` at the root level.

**How to run**:
```bash
# Full rebuild
python rebuild_pipeline.py

# Debug specific QB
python rebuild_pipeline.py -q "Stafford"

# All same arguments
python rebuild_pipeline.py -q "Stafford" --stage payment
```

**Implementation**: `rebuild_pipeline.py` would be:
```python
#!/usr/bin/env python3
"""
QB Data Pipeline - Main entry point
"""
from qb_research.preprocessing.pipeline import main

if __name__ == "__main__":
    main()
```

**Pros**: 
- ✅ Clean, obvious name
- ✅ Follows pattern of other scripts (`qb_comp_tool.py`, `standings_scraper.py`)
- ✅ Can eventually deprecate `QB_research.py`

**Cons**: 
- ⚠️ Need to update any scripts/docs that reference `QB_research.py`

---

### Option 3: Run as Python Module (Most Pythonic)

**What happens**: Use Python's `-m` flag to run the module directly.

**How to run**:
```bash
# Full rebuild
python -m qb_research.preprocessing.pipeline

# Debug specific QB
python -m qb_research.preprocessing.pipeline -q "Stafford"

# All same arguments
python -m qb_research.preprocessing.pipeline -q "Stafford" --stage payment
```

**Implementation**: `qb_research/preprocessing/pipeline.py` would have:
```python
def main():
    """Entry point with argparse"""
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()
    rebuild_pipeline(...)

if __name__ == "__main__":
    main()
```

**Pros**: 
- ✅ Most "Pythonic" approach
- ✅ Clear that it's part of the package
- ✅ Works well with package structure

**Cons**: 
- ⚠️ Longer command to type
- ⚠️ Less obvious for non-Python developers

---

## Recommended Approach: Hybrid

**Use both Option 1 and Option 2**:

1. **Keep `QB_research.py`** as backward compatibility (for existing workflows)
2. **Create `rebuild_pipeline.py`** as the new recommended way (for new users/docs)

This gives you:
- ✅ Existing scripts continue working
- ✅ New clean entry point
- ✅ Can migrate gradually
- ✅ Can deprecate `QB_research.py` later

### File Structure

```
football/
├── QB_research.py              # Backward compatibility (thin wrapper)
├── rebuild_pipeline.py          # New recommended entry point
├── qb_research/
│   └── preprocessing/
│       └── pipeline.py          # Actual implementation
```

### Implementation

**`rebuild_pipeline.py`** (new, recommended):
```python
#!/usr/bin/env python3
"""
QB Data Pipeline - Main entry point

Usage:
    python rebuild_pipeline.py                    # Full rebuild
    python rebuild_pipeline.py -q "Stafford"      # Debug QB
    python rebuild_pipeline.py -q "Stafford" --stage payment
"""
from qb_research.preprocessing.pipeline import main

if __name__ == "__main__":
    main()
```

**`QB_research.py`** (backward compatibility):
```python
"""
QB Research - Backward compatibility wrapper

DEPRECATED: Use 'python rebuild_pipeline.py' instead.
This file is kept for backward compatibility only.
"""
import warnings
from qb_research.preprocessing.pipeline import main

if __name__ == "__main__":
    warnings.warn(
        "QB_research.py is deprecated. Use 'python rebuild_pipeline.py' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    main()
```

**`qb_research/preprocessing/pipeline.py`** (actual implementation):
```python
"""
Pipeline orchestration and rebuild workflow
"""
import argparse
from qb_research.data.builders import fix_individual_qb_files, create_all_seasons_from_existing_qb_files
from qb_research.data.mappers import create_contract_player_mapping
# ... other imports

def rebuild_pipeline(debug_qb=None, stage='all', skip_rebuild=False):
    """
    Main rebuild workflow - orchestrates the entire data pipeline.
    
    Args:
        debug_qb: Optional QB name to debug throughout pipeline
        stage: Which pipeline stage to check ('all', 'contracts', 'payment', etc.)
        skip_rebuild: If True, skip rebuild and just run QB debug
    """
    # All the workflow logic from current __main__ block
    print("="*80)
    print("QB DATA REBUILD WORKFLOW")
    print("="*80)
    
    if debug_qb and skip_rebuild:
        from qb_research.comparisons.comp_analysis import debug_specific_qb
        debug_specific_qb(debug_qb, stage)
        return
    
    # STEP 1: Fix individual QB files
    fix_individual_qb_files()
    
    # ... rest of pipeline steps
    # ...

def main():
    """Entry point with argparse"""
    parser = argparse.ArgumentParser(description='QB Data Pipeline and Analysis')
    parser.add_argument('-q', '--qb', type=str, help='Debug specific QB (e.g., "Stafford")')
    parser.add_argument('--stage', type=str, default='all', 
                       choices=['contracts', 'payment', 'prepared', 'era', 'export', 'all'],
                       help='Which pipeline stage to check')
    parser.add_argument('--skip-rebuild', action='store_true', 
                       help='Skip full rebuild, just run QB debug')
    
    args = parser.parse_args()
    rebuild_pipeline(
        debug_qb=args.qb,
        stage=args.stage,
        skip_rebuild=args.skip_rebuild
    )

if __name__ == "__main__":
    main()
```

---

## Quick Reference

### Recommended Commands (After Refactoring)

```bash
# Full rebuild (new way)
python rebuild_pipeline.py

# Full rebuild (old way - still works)
python QB_research.py

# Debug QB (new way)
python rebuild_pipeline.py -q "Stafford"

# Debug QB with stage (new way)
python rebuild_pipeline.py -q "Stafford" --stage payment

# Skip rebuild, just debug (new way)
python rebuild_pipeline.py -q "Stafford" --skip-rebuild
```

### VS Code Terminal

In VS Code, you can run these commands directly in the integrated terminal:

1. Open terminal: `` Ctrl+` `` (or View → Terminal)
2. Make sure you're in the project root directory
3. Run any of the commands above

### VS Code Tasks (Optional)

You can also create VS Code tasks for common operations. Create `.vscode/tasks.json`:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Rebuild Pipeline",
            "type": "shell",
            "command": "python rebuild_pipeline.py",
            "group": "build",
            "presentation": {
                "reveal": "always",
                "panel": "new"
            }
        },
        {
            "label": "Debug QB",
            "type": "shell",
            "command": "python rebuild_pipeline.py -q ${input:qbName}",
            "group": "build",
            "presentation": {
                "reveal": "always",
                "panel": "new"
            }
        }
    ],
    "inputs": [
        {
            "id": "qbName",
            "type": "promptString",
            "description": "Enter QB name to debug"
        }
    ]
}
```

Then you can run tasks via: `Ctrl+Shift+P` → "Tasks: Run Task" → "Rebuild Pipeline"

---

## Summary

**Best approach**: Use **Option 1 + Option 2 (Hybrid)**
- Keep `QB_research.py` working for backward compatibility
- Create `rebuild_pipeline.py` as the new recommended way
- Both call the same underlying function in `qb_research/preprocessing/pipeline.py`

**Commands stay the same** - you can use either:
- `python QB_research.py` (old way, still works)
- `python rebuild_pipeline.py` (new way, recommended)

Both do exactly the same thing!

