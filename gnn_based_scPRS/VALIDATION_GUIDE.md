# GNN Project Validation Guide

This guide outlines the validation steps for the Single Cell Polygenic Risk Score (scPRS) GNN project.

## Quick Validation

Run the quick validation script to get an immediate overview:

```bash
python3 quick_validate.py
```

This checks for:
- ✅ Project structure (data/, scripts/, models/, results/)
- ✅ Key data files (PRS data, gene networks, single-cell data)
- ✅ Trained model files
- ✅ Core scripts
- ✅ Results directories
- ✅ Documentation files

## Comprehensive Validation

For detailed validation, run the full validation suite:

```bash
# Activate virtual environment first
source scprs_env/bin/activate
python validate_gnn_project.py
```

### Validation Components

#### 1. Environment Validation
- **PyTorch**: Core deep learning framework
- **PyTorch Geometric**: Graph neural network library
- **Scanpy/AnnData**: Single-cell data handling
- **Scikit-learn**: Machine learning utilities
- **CUDA**: GPU acceleration availability

#### 2. Data Integrity Validation
- **GWAS data**: Format and required columns (SNP, CHR, BP, BETA, P)
- **Gene coordinates**: Genomic positions and annotations
- **Gene networks**: Protein-protein interaction data
- **Single-cell data**: H5AD format validation
- **PRS scores**: Gene-level and cell-type PRS data

#### 3. Model Validation
- **GNN model loading**: Can trained models be loaded successfully?
- **Model architecture**: Basic forward pass functionality
- **Gene embeddings**: Pre-trained embeddings availability

#### 4. Pipeline Execution Validation
- **PRS calculation**: Core computation functions
- **GNN training**: Model instantiation and basic operations
- **Data processing**: File I/O and preprocessing

#### 5. Results Validation
- **Target prioritization**: Top gene rankings
- **Cell-type enrichment**: CELLECT results
- **Cell specificity**: CELLEX analysis outputs

#### 6. Biological Plausibility
- **PRS distribution**: Statistical properties of risk scores
- **Cell type coverage**: Number of cell types analyzed
- **Value ranges**: Reasonable bounds for biological scores

#### 7. Cross-Validation
- **Performance metrics**: Model evaluation on held-out data
- **Stability**: Consistency across different data splits

## Expected Validation Results

### ✅ Excellent Status
- All tests pass
- No critical errors
- ≤2 warnings

### ✅ Good Status
- All tests pass
- Some minor warnings (data file issues, optional dependencies)

### ⚠️ Needs Attention
- Few failures (≤3)
- Addressable issues (missing files, format problems)

### ❌ Critical Issues
- Multiple failures (>3)
- Core functionality broken
- Environment/setup problems

## Common Issues and Solutions

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### Data File Issues
- Check file paths in scripts
- Verify data formats match expectations
- Ensure GWAS files have required columns

### Model Loading Errors
- Check PyTorch version compatibility
- Verify model save/load format
- Ensure CUDA availability if needed

### Pipeline Execution Failures
- Check data dependencies between scripts
- Verify intermediate file generation
- Ensure proper file permissions

## Validation Output Interpretation

The validation script provides detailed feedback:

```
✓ Environment: PyTorch version 2.0.1
✓ Data integrity: gene_level_prs.csv - Shape: (15000, 3)
⚠️ CUDA: Available: False (consider GPU acceleration)
✗ Model loading: prs_gnn_model.pt - Error: incompatible architecture
```

## Next Steps After Validation

### If Validation Passes ✅
1. **Run full pipeline**: Execute end-to-end analysis
2. **Generate figures**: Create publication-ready plots
3. **Performance benchmarking**: Compare with baseline methods
4. **Sensitivity analysis**: Test parameter robustness

### If Issues Found ⚠️
1. **Fix critical errors**: Address environment/data issues
2. **Update documentation**: Reflect any changes made
3. **Re-run validation**: Confirm fixes work
4. **Version control**: Commit fixes to git

### Advanced Validation 🧪
1. **Biological validation**: Compare with known disease genes
2. **Cross-dataset validation**: Test on independent datasets
3. **Method comparison**: Benchmark against other PRS methods
4. **Clinical validation**: Correlate with real clinical outcomes

## Automated Validation

For continuous integration, add to your workflow:

```yaml
# .github/workflows/validate.yml
name: Validate GNN Project
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run validation
      run: python validate_gnn_project.py
```

## Performance Metrics to Track

- **Model accuracy**: Cross-validation performance
- **Biological relevance**: Enrichment in known disease genes
- **Computational efficiency**: Runtime and memory usage
- **Reproducibility**: Consistency across runs
- **Scalability**: Performance with larger datasets

## Troubleshooting

### Validation Script Won't Run
- Check Python version (3.9+ required)
- Ensure virtual environment is activated
- Install missing dependencies

### Data Loading Errors
- Verify file paths are correct
- Check file permissions
- Ensure data formats match expectations

### Model Issues
- Check PyTorch Geometric compatibility
- Verify CUDA installation if using GPU
- Ensure model files aren't corrupted

For additional help, check the project issues or create a new issue with validation output.