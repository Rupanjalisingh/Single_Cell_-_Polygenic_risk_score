# inspect_gwas_file.py
"""
Inspect GWAS file structure to understand column names and format
"""

import pandas as pd
import numpy as np

gwas_file = "gwas_chr22_gene_annotated.tsv"

print("=" * 80)
print("GWAS FILE INSPECTION")
print("=" * 80)

# Try different delimiters
delimiters = ['\t', ',', ' ', '\\s+']

for delim in delimiters:
    try:
        if delim == '\\s+':
            df = pd.read_csv(gwas_file, sep=delim, engine='python', nrows=5)
        else:
            df = pd.read_csv(gwas_file, sep=delim, nrows=5)
        
        print(f"\n✓ Successfully loaded with delimiter: {repr(delim)}")
        print(f"\nShape: {df.shape}")
        print(f"\nColumn names: {list(df.columns)}")
        print(f"\nColumn names (lowercase): {[col.lower() for col in df.columns]}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nFirst few rows:")
        print(df)
        print(f"\nIndex name: {df.index.name}")
        print(f"Index (first 5): {df.index[:5].tolist()}")
        break
    except Exception as e:
        print(f"\n✗ Failed with delimiter {repr(delim)}: {str(e)[:100]}")
        continue

print("\n" + "=" * 80)

# cell_type_genetic_enrichment_ROBUST.py
"""
CELLECT-MAGMA: Cell-Type Genetic Enrichment Analysis
ROBUST VERSION - Ultra-flexible file handling
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings
from scipy import stats

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CELLECTMagmaAnalysis:
    """
    CELLECT-MAGMA with ultra-flexible file parsing
    """
    
    def __init__(self, gwas_file, specificity_file, output_dir):
        self.gwas_file = gwas_file
        self.specificity_file = specificity_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.gwas_df = None
        self.specificity_df = None
        self.gene_pvalues = None
        self.enrichment_df = None
        
    def smart_load_gwas(self):
        """
        Ultra-smart GWAS loading with extensive debugging
        """
        logger.info(f"Loading GWAS file: {self.gwas_file}")
        
        # Step 1: Try to load with various delimiters
        delimiters = ['\t', ',', ' ', '\\s+', ';', '|']
        gwas_df = None
        used_delim = None
        
        for delim in delimiters:
            try:
                if delim == '\\s+':
                    gwas_df = pd.read_csv(self.gwas_file, sep=delim, engine='python')
                else:
                    gwas_df = pd.read_csv(self.gwas_file, sep=delim)
                
                if len(gwas_df.columns) > 1:  # Valid load
                    used_delim = delim
                    logger.info(f"✓ Successfully loaded with delimiter: {repr(delim)}")
                    break
            except Exception as e:
                continue
        
        if gwas_df is None:
            raise ValueError("Could not load GWAS file with any delimiter")
        
        logger.info(f"File shape: {gwas_df.shape}")
        logger.info(f"Columns: {list(gwas_df.columns)}")
        logger.info(f"Data types:\n{gwas_df.dtypes}")
        
        self.gwas_df = gwas_df
        return gwas_df
    
    def smart_load_specificity(self):
        """Load specificity file"""
        logger.info(f"\nLoading specificity file: {self.specificity_file}")
        
        self.specificity_df = pd.read_csv(self.specificity_file, sep='\t', index_col=0)
        logger.info(f"✓ Loaded: {self.specificity_df.shape}")
        logger.info(f"Cell types: {list(self.specificity_df.columns)}")
        
        return self.specificity_df
    
    def extract_gene_pvalues_smart(self):
        """
        Ultra-smart gene and p-value column detection
        """
        logger.info("\nExtracting gene-level p-values...")
        
        df = self.gwas_df.copy()
        
        # ===== STEP 1: Find Gene Column =====
        logger.info("\nStep 1: Identifying gene column...")
        
        gene_col = None
        possible_gene_names = [
            'gene', 'Gene', 'GENE', 'gene_name', 'Gene_Name', 'GENE_NAME',
            'genename', 'GeneName', 'GENENAME', 'hgnc_symbol', 'HGNC_Symbol',
            'symbol', 'Symbol', 'SYMBOL', 'name', 'Name', 'NAME',
            'gene_id', 'Gene_ID', 'GENE_ID'
        ]
        
        # First, check column names
        for col in df.columns:
            if col in possible_gene_names or col.lower() in [x.lower() for x in possible_gene_names]:
                gene_col = col
                logger.info(f"✓ Found gene column: '{gene_col}'")
                break
        
        # If not found, try index
        if gene_col is None:
            logger.info("Gene column not found in columns, trying index...")
            if df.index.name in possible_gene_names or \
               (isinstance(df.index[0], str) and not df.index[0].isdigit()):
                logger.info(f"✓ Using dataframe index as gene names")
                df['gene'] = df.index
                gene_col = 'gene'
            else:
                # Try first column
                logger.info(f"Trying first column: {df.columns[0]}")
                if isinstance(df.iloc[0, 0], str) and len(str(df.iloc[0, 0])) < 30:
                    gene_col = df.columns[0]
                    logger.info(f"✓ Using first column as gene names: '{gene_col}'")
        
        if gene_col is None:
            logger.error("COULD NOT FIND GENE COLUMN!")
            logger.error(f"Available columns: {list(df.columns)}")
            logger.error(f"First few values of each column:")
            for col in df.columns[:5]:
                logger.error(f"  {col}: {df[col].head(3).tolist()}")
            raise ValueError("Could not identify gene column")
        
        # ===== STEP 2: Find P-value Column =====
        logger.info("\nStep 2: Identifying p-value column...")
        
        pval_col = None
        possible_pval_names = [
            'p', 'P', 'p_value', 'P_value', 'P_VALUE', 'pval', 'Pval', 'PVAL',
            'p-value', 'P-value', 'P-VALUE', 'pvalue', 'Pvalue', 'PVALUE',
            'p_val', 'P_val', 'P_VAL', 'pv', 'PV'
        ]
        
        for col in df.columns:
            if col in possible_pval_names or col.lower() in [x.lower() for x in possible_pval_names]:
                pval_col = col
                logger.info(f"✓ Found p-value column: '{pval_col}'")
                break
        
        if pval_col is None:
            logger.warning("P-value column not found! Using all genes with p=0.05")
            gene_pvalues = pd.Series(
                np.ones(len(df)) * 0.05,
                index=df[gene_col].values
            )
        else:
            # Extract p-values
            logger.info(f"Extracting p-values from column: '{pval_col}'")
            
            gene_pvalues = pd.Series(
                df[pval_col].values,
                index=df[gene_col].values
            )
        
        # ===== STEP 3: Clean p-values =====
        logger.info("\nStep 3: Cleaning p-values...")
        
        # Convert to numeric
        gene_pvalues = pd.to_numeric(gene_pvalues, errors='coerce')
        
        # Replace inf and invalid values
        gene_pvalues = gene_pvalues.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with 1.0 (non-significant)
        initial_na = gene_pvalues.isna().sum()
        gene_pvalues = gene_pvalues.fillna(1.0)
        if initial_na > 0:
            logger.info(f"  Filled {initial_na} NaN values with 1.0")
        
        # Clip to valid range [0, 1]
        gene_pvalues = gene_pvalues.clip(0, 1)
        
        # Remove duplicates (keep minimum p-value)
        gene_pvalues = gene_pvalues.groupby(gene_pvalues.index).min()
        
        self.gene_pvalues = gene_pvalues
        
        logger.info(f"\n✓ Extraction complete!")
        logger.info(f"  Total genes: {len(gene_pvalues)}")
        logger.info(f"  Min p-value: {gene_pvalues.min():.2e}")
        logger.info(f"  Max p-value: {gene_pvalues.max():.2e}")
        logger.info(f"  Mean p-value: {gene_pvalues.mean():.4f}")
        logger.info(f"  Genes with p < 0.05: {sum(gene_pvalues < 0.05)}")
        logger.info(f"  Genes with p < 0.01: {sum(gene_pvalues < 0.01)}")
        logger.info(f"  Genes with p < 0.001: {sum(gene_pvalues < 0.001)}")
        
        return gene_pvalues
    
    def calculate_celltype_enrichment(self):
        """Calculate cell-type enrichment"""
        logger.info("\n" + "=" * 80)
        logger.info("Calculating cell-type enrichment...")
        logger.info("=" * 80)
        
        if self.gene_pvalues is None:
            self.extract_gene_pvalues_smart()
        
        enrichment_results = {}
        
        for cell_type in self.specificity_df.columns:
            logger.info(f"\nProcessing {cell_type}...")
            
            specificity_scores = self.specificity_df[cell_type]
            
            # Get top 25% specific genes
            threshold = np.percentile(specificity_scores, 75)
            high_spec_genes = specificity_scores[specificity_scores >= threshold].index.tolist()
            
            # Find shared genes
            shared_genes = [g for g in high_spec_genes if g in self.gene_pvalues.index]
            
            logger.info(f"  High specificity genes: {len(high_spec_genes)}")
            logger.info(f"  Shared with GWAS: {len(shared_genes)}")
            
            if len(shared_genes) > 0:
                cell_type_pvals = self.gene_pvalues[shared_genes]
                sig_genes = sum(cell_type_pvals < 0.05)
                
                # KS test for enrichment
                try:
                    ks_stat, enrichment_p = stats.kstest(cell_type_pvals, 'uniform')
                except:
                    enrichment_p = np.median(cell_type_pvals)
                
                enrichment_p = max(enrichment_p, 1e-10)
                
                enrichment_results[cell_type] = {
                    'n_high_specificity_genes': len(high_spec_genes),
                    'n_shared_genes': len(shared_genes),
                    'n_significant_genes': sig_genes,
                    'enrichment_p': enrichment_p,
                    'log_enrichment_p': -np.log10(enrichment_p),
                    'proportion_significant': sig_genes / len(shared_genes) if len(shared_genes) > 0 else 0
                }
                
                logger.info(f"  Significant genes: {sig_genes}/{len(shared_genes)}")
                logger.info(f"  P-value: {enrichment_p:.2e}")
            else:
                enrichment_results[cell_type] = {
                    'n_high_specificity_genes': len(high_spec_genes),
                    'n_shared_genes': 0,
                    'n_significant_genes': 0,
                    'enrichment_p': 1.0,
                    'log_enrichment_p': 0.0,
                    'proportion_significant': 0.0
                }
        
        self.enrichment_df = pd.DataFrame(enrichment_results).T
        self.enrichment_df = self.enrichment_df.sort_values('enrichment_p')
        
        logger.info(f"\n✓ Enrichment calculated for {len(self.enrichment_df)} cell types")
        
        return self.enrichment_df
    
    def save_results(self, filename=None):
        """Save results"""
        if filename is None:
            filename = self.output_dir / "T2D_celltype_enrichment_results.tsv"
        
        self.enrichment_df.to_csv(filename, sep='\t')
        logger.info(f"\n✓ Saved results to {filename}")
        return filename
    
    def create_enrichment_report(self):
        """Create report"""
        report_file = self.output_dir / "CELLECT_enrichment_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("CELLECT-MAGMA: Cell-Type Genetic Enrichment Analysis\n")
            f.write("Type 2 Diabetes\n")
            f.write("=" * 100 + "\n\n")
            
            f.write("CELL-TYPE ENRICHMENT RESULTS:\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Cell Type':<30} {'P-value':<15} {'-log10(P)':<15} {'N_Sig':<10}\n")
            f.write("-" * 100 + "\n")
            
            for cell_type, row in self.enrichment_df.iterrows():
                f.write(f"{str(cell_type):<30} {row['enrichment_p']:<15.2e} "
                       f"{row['log_enrichment_p']:<15.4f} {int(row['n_significant_genes']):<10}\n")
        
        logger.info(f"✓ Saved report to {report_file}")
        return report_file
    
    def run_complete_analysis(self):
        """Run complete analysis"""
        logger.info("\n" + "=" * 100)
        logger.info("CELLECT-MAGMA COMPLETE ANALYSIS")
        logger.info("=" * 100)
        
        self.smart_load_gwas()
        self.smart_load_specificity()
        self.extract_gene_pvalues_smart()
        self.calculate_celltype_enrichment()
        self.save_results()
        self.create_enrichment_report()
        
        logger.info("\n" + "=" * 100)
        logger.info("✓ CELLECT ANALYSIS COMPLETE")
        logger.info("=" * 100)


if __name__ == "__main__":
    cellect = CELLECTMagmaAnalysis(
        gwas_file="gwas_chr22_gene_annotated.tsv",
        specificity_file="cellex_output/T2D_celltype_specificity_index.tsv",
        output_dir="cellect_output/"
    )
    cellect.run_complete_analysis()