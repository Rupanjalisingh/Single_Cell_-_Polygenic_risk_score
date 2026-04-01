#!/usr/bin/env python3
"""
Quick Validation Script for GNN Project
Run this to get a fast overview of project status
"""

import os
import sys
from pathlib import Path

def quick_validate():
    """Quick validation checks"""
    project_root = Path.cwd()

    checks = {
        "Project structure": False,
        "Data files": False,
        "Model files": False,
        "Scripts": False,
        "Results": False,
        "Documentation": False
    }

    # Check directories
    required_dirs = ["data", "scripts", "models", "results"]
    checks["Project structure"] = all((project_root / d).exists() for d in required_dirs)

    # Check key data files
    key_data_files = ["gene_level_prs.csv", "string_edges.csv", "type_2_diabetes_pancreas.h5ad"]
    checks["Data files"] = all((project_root / "data" / f).exists() for f in key_data_files)

    # Check model files
    model_files = ["prs_gnn_model.pt", "gene_embeddings.pt"]
    checks["Model files"] = all((project_root / "models" / f).exists() for f in model_files)

    # Check scripts
    key_scripts = ["gnn_model.py", "gnn_run.py", "prs_calculation.py"]
    checks["Scripts"] = all((project_root / "scripts" / f).exists() for f in key_scripts)

    # Check results
    result_dirs = ["targets_output", "cellect_output", "cellex_output"]
    checks["Results"] = all((project_root / "results" / d).exists() for d in result_dirs)

    # Check documentation
    doc_files = ["README.md", "requirements.txt", "LICENSE"]
    checks["Documentation"] = all((project_root / f).exists() for f in doc_files)

    # Print results
    print("🚀 GNN Project Quick Validation")
    print("=" * 40)

    all_passed = True
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 ALL CHECKS PASSED - Project is ready!")
        return True
    else:
        print("⚠️  Some checks failed - Review and fix issues")
        return False

if __name__ == "__main__":
    success = quick_validate()
    sys.exit(0 if success else 1)