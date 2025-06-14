# 🛠️ Infera – Pre-Deployment Cleanup Checklist

This document tracks technical workarounds and temporary scaffolding decisions that must be resolved before Infera is packaged or deployed as a production-grade tool.

---

### ✅ 1. Temporary Import Bootstrapping in `sec_fetcher.py`

**📍 Location:** `ingest/sec_fetcher.py`  
**🛠 Current Fix:**
```bash
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```

 🔍 Why It Exists:
This line manually appends the project root directory to sys.path to allow importing infera_config.py during local script execution (e.g., python ingest/sec_fetcher.py).

🚫 Problem:
This is a non-standard workaround that won’t scale well or function properly in packaged or deployed environments.

✅ Pre-Deployment Fix:

- Restructure Infera as a proper Python package using __init__.py
- Replace with standard module-based imports
- Use command-line execution via:
  python -m ingest.sec_fetcher

or the fully namespaced form if packaged:

```bash
python -m infera.ingest.sec_fetcher
```