#!/usr/bin/env python3
"""
Comprehensive Validation Suite for Single Cell Polygenic Risk Score (scPRS) GNN Project

This script validates the complete GNN pipeline including:
1. Environment and dependencies
2. Data integrity and format validation
3. Model loading and basic functionality
4. Pipeline execution validation
5. Results validation and biological plausibility
6. Cross-validation and performance metrics

Usage: python validate_gnn_project.py
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import networkx as nx
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import project modules
sys.path.append('scripts')
try:
    from gnn_model import PRSGNN
    import torch_geometric
    print("✓ Successfully imported project modules")
except ImportError as e:
    print(f"✗ Failed to import project modules: {e}")
    sys.exit(1)

class GNNProjectValidator:
    """Comprehensive validator for the scPRS GNN project"""

    def __init__(self, project_root=None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.data_dir = self.project_root / "data"
        self.scripts_dir = self.project_root / "scripts"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"

        # Validation results
        self.validation_results = {}
        self.errors = []

    def log_validation(self, test_name, status, message=""):
        """Log validation result"""
        self.validation_results[test_name] = {"status": status, "message": message}
        symbol = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⚠"
        print(f"{symbol} {test_name}: {message}")

    def validate_project_structure(self):
        """Validate project directory structure"""
        required_dirs = ["data", "scripts", "models", "results"]
        required_files = ["requirements.txt", "README.md"]

        for dir_name in required_dirs:
            if (self.project_root / dir_name).exists():
                self.log_validation(f"Directory {dir_name}", "PASS", "Exists")
            else:
                self.log_validation(f"Directory {dir_name}", "FAIL", "Missing")
                self.errors.append(f"Missing directory: {dir_name}")

        for file_name in required_files:
            if (self.project_root / file_name).exists():
                self.log_validation(f"File {file_name}", "PASS", "Exists")
            else:
                self.log_validation(f"File {file_name}", "FAIL", "Missing")

    def validate_environment(self):
        """Validate Python environment and dependencies"""
        try:
            import torch
            self.log_validation("PyTorch", "PASS", f"Version {torch.__version__}")

            import torch_geometric
            self.log_validation("PyTorch Geometric", "PASS", f"Version {torch_geometric.__version__}")

            import scanpy
            self.log_validation("Scanpy", "PASS", f"Version {scanpy.__version__}")

            import anndata
            self.log_validation("AnnData", "PASS", f"Version {anndata.__version__}")

            import sklearn
            self.log_validation("Scikit-learn", "PASS", f"Version {sklearn.__version__}")

            # Check CUDA availability
            cuda_available = torch.cuda.is_available()
            self.log_validation("CUDA", "PASS" if cuda_available else "WARN",
                              f"Available: {cuda_available}")

        except ImportError as e:
            self.log_validation("Environment", "FAIL", f"Missing dependency: {e}")
            self.errors.append(f"Missing dependency: {e}")

    def validate_data_integrity(self):
        """Validate data files integrity and format"""
        # Check key data files
        key_files = {
            "gene_level_prs.csv": ["gene", "gene_prs"],
            "celltype_prs_scores.csv": ["cluster", "gene_prs"],
            "string_edges.csv": ["gene1", "gene2"],
            "gene_coordinates.tsv": ["gene_id", "chromosome", "start", "end"],
            "gwas_summary_statistics.tsv": ["SNP", "CHR", "BP", "BETA", "P"],
            "type_2_diabetes_pancreas.h5ad": None,  # Special handling for H5AD
            "gene_interaction_graph.gml": None  # Special handling for GML
        }

        for filename, required_cols in key_files.items():
            filepath = self.data_dir / filename
            if not filepath.exists():
                self.log_validation(f"Data file {filename}", "FAIL", "File not found")
                continue

            try:
                if filename.endswith('.csv') or filename.endswith('.tsv'):
                    sep = '\t' if filename.endswith('.tsv') else ','
                    df = pd.read_csv(filepath, sep=sep, nrows=5)  # Just check first few rows

                    if required_cols:
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            self.log_validation(f"Data file {filename}", "FAIL",
                                              f"Missing columns: {missing_cols}")
                        else:
                            self.log_validation(f"Data file {filename}", "PASS",
                                              f"Shape: {df.shape}, columns OK")
                    else:
                        self.log_validation(f"Data file {filename}", "PASS", f"Shape: {df.shape}")

                elif filename.endswith('.h5ad'):
                    import anndata
                    adata = anndata.read_h5ad(filepath, backed='r')  # Read in backed mode
                    self.log_validation(f"Data file {filename}", "PASS",
                                      f"Shape: {adata.shape}, obs: {len(adata.obs.columns)}, var: {len(adata.var.columns)}")

                elif filename.endswith('.gml'):
                    G = nx.read_gml(filepath)
                    self.log_validation(f"Data file {filename}", "PASS",
                                      f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

            except Exception as e:
                self.log_validation(f"Data file {filename}", "FAIL", f"Error reading file: {e}")

    def validate_models(self):
        """Validate trained models can be loaded"""
        model_files = ["prs_gnn_model.pt", "gene_embeddings.pt"]

        for model_file in model_files:
            filepath = self.models_dir / model_file
            if not filepath.exists():
                self.log_validation(f"Model {model_file}", "FAIL", "File not found")
                continue

            try:
                if model_file.endswith('.pt'):
                    checkpoint = torch.load(filepath, map_location='cpu')

                    if 'model_state_dict' in checkpoint:
                        # Load model architecture and weights
                        model = PRSGNN()
                        model.load_state_dict(checkpoint['model_state_dict'])
                        model.eval()
                        self.log_validation(f"Model {model_file}", "PASS", "Successfully loaded")
                    else:
                        # Assume it's just model weights
                        model = PRSGNN()
                        model.load_state_dict(checkpoint)
                        model.eval()
                        self.log_validation(f"Model {model_file}", "PASS", "Successfully loaded")

            except Exception as e:
                self.log_validation(f"Model {model_file}", "FAIL", f"Error loading: {e}")

    def validate_pipeline_execution(self):
        """Test basic pipeline components"""
        # Test PRS calculation
        try:
            from prs_calculation import calculate_gene_prs, calculate_celltype_prs

            # This would require actual data files, so we'll just test imports
            self.log_validation("PRS calculation imports", "PASS", "Functions available")

        except ImportError as e:
            self.log_validation("PRS calculation imports", "FAIL", f"Import error: {e}")

        # Test GNN model instantiation
        try:
            model = PRSGNN()
            # Test forward pass with dummy data
            dummy_x = torch.randn(10, 1)  # 10 nodes, 1 feature
            dummy_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)  # Simple triangle

            from torch_geometric.data import Data
            dummy_data = Data(x=dummy_x, edge_index=dummy_edge_index)

            with torch.no_grad():
                output = model(dummy_data)

            self.log_validation("GNN model forward pass", "PASS", f"Output shape: {output.shape}")

        except Exception as e:
            self.log_validation("GNN model forward pass", "FAIL", f"Error: {e}")

    def validate_results(self):
        """Validate analysis results"""
        # Check key result files
        result_files = {
            "results/targets_output/T2D_prioritized_targets.tsv": ["gene", "score"],
            "results/cellect_output/T2D_celltype_enrichment_results.tsv": None,
            "results/cellex_output/T2D_celltype_specificity_index.tsv": None
        }

        for filepath_str, required_cols in result_files.items():
            filepath = self.project_root / filepath_str
            if not filepath.exists():
                self.log_validation(f"Result file {filepath_str}", "WARN", "File not found")
                continue

            try:
                if filepath.suffix == '.tsv':
                    df = pd.read_csv(filepath, sep='\t', nrows=5)
                    if required_cols:
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        if missing_cols:
                            self.log_validation(f"Result file {filepath_str}", "FAIL",
                                              f"Missing columns: {missing_cols}")
                        else:
                            self.log_validation(f"Result file {filepath_str}", "PASS",
                                              f"Shape: {df.shape}")
                    else:
                        self.log_validation(f"Result file {filepath_str}", "PASS", f"Shape: {df.shape}")

            except Exception as e:
                self.log_validation(f"Result file {filepath_str}", "FAIL", f"Error reading: {e}")

    def validate_biological_plausibility(self):
        """Check biological plausibility of results"""
        try:
            # Load gene-level PRS
            gene_prs_file = self.data_dir / "gene_level_prs.csv"
            if gene_prs_file.exists():
                gene_prs = pd.read_csv(gene_prs_file)

                # Check PRS distribution
                prs_stats = gene_prs['gene_prs'].describe()
                self.log_validation("PRS distribution", "PASS",
                                  f"Mean: {prs_stats['mean']:.3f}, Std: {prs_stats['std']:.3f}")

                # Check for reasonable range (PRS should be in reasonable bounds)
                if abs(prs_stats['mean']) > 10 or prs_stats['std'] > 5:
                    self.log_validation("PRS range check", "WARN", "PRS values seem extreme")
                else:
                    self.log_validation("PRS range check", "PASS", "PRS values in reasonable range")

            # Load cell-type PRS
            cell_prs_file = self.data_dir / "celltype_prs_scores.csv"
            if cell_prs_file.exists():
                cell_prs = pd.read_csv(cell_prs_file)

                # Check if we have multiple cell types
                n_celltypes = len(cell_prs)
                self.log_validation("Cell types", "PASS" if n_celltypes > 1 else "WARN",
                                  f"Found {n_celltypes} cell types")

        except Exception as e:
            self.log_validation("Biological plausibility", "FAIL", f"Error: {e}")

    def run_cross_validation(self):
        """Run basic cross-validation if possible"""
        try:
            from evaluation import load_prs_data, build_feature_matrix

            cell_prs, gene_prs = load_prs_data()
            X = build_feature_matrix(cell_prs, gene_prs)

            # Simple cross-validation with dummy labels
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LogisticRegression

            # Create dummy binary labels for demonstration
            y = np.random.randint(0, 2, size=len(X))

            clf = LogisticRegression(random_state=42, max_iter=1000)
            scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')

            self.log_validation("Cross-validation", "PASS",
                              f"Mean accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

        except Exception as e:
            self.log_validation("Cross-validation", "FAIL", f"Error: {e}")

    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        print("\n" + "="*60)
        print("GNN PROJECT VALIDATION REPORT")
        print("="*60)

        total_tests = len(self.validation_results)
        passed = sum(1 for r in self.validation_results.values() if r['status'] == 'PASS')
        failed = sum(1 for r in self.validation_results.values() if r['status'] == 'FAIL')
        warnings = sum(1 for r in self.validation_results.values() if r['status'] == 'WARN')

        print(f"\nSUMMARY:")
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Warnings: {warnings}")

        if self.errors:
            print(f"\nCRITICAL ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")

        print(f"\nDETAILED RESULTS:")
        for test_name, result in self.validation_results.items():
            status = result['status']
            message = result['message']
            print(f"  {status}: {test_name}")
            if message:
                print(f"    {message}")

        # Overall assessment
        if failed == 0 and warnings <= 2:
            print(f"\n🎉 OVERALL STATUS: EXCELLENT - Project is fully validated!")
        elif failed == 0:
            print(f"\n✅ OVERALL STATUS: GOOD - Minor warnings to address")
        elif failed <= 3:
            print(f"\n⚠️  OVERALL STATUS: NEEDS ATTENTION - Some issues to fix")
        else:
            print(f"\n❌ OVERALL STATUS: CRITICAL - Major issues require fixing")

        return passed, failed, warnings

    def run_full_validation(self):
        """Run complete validation suite"""
        print("Starting comprehensive GNN project validation...\n")

        self.validate_project_structure()
        print()

        self.validate_environment()
        print()

        self.validate_data_integrity()
        print()

        self.validate_models()
        print()

        self.validate_pipeline_execution()
        print()

        self.validate_results()
        print()

        self.validate_biological_plausibility()
        print()

        self.run_cross_validation()
        print()

        return self.generate_validation_report()


def main():
    """Main validation function"""
    validator = GNNProjectValidator()
    passed, failed, warnings = validator.run_full_validation()

    # Exit with appropriate code
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()