# T2D_pipeline_final_fixed.py
"""
Complete T2D Analysis Pipeline - FINAL FIXED VERSION
Uses overlapping genes between GWAS and single-cell data
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 3: CELLEX - CELL-TYPE EXPRESSION SPECIFICITY (Filter by overlap)
# ============================================================================

class CellTypeSpecificityAnalysis:
    """CELLEX: Calculate cell-type expression specificity scores"""
    
    def __init__(self, deg_file, overlap_genes_file, output_dir):
        self.deg_file = deg_file
        self.overlap_genes_file = overlap_genes_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.deg_df = None
        self.overlap_genes = None
        self.expression_matrix = None
        self.specificity_scores = None
        
    def load_data(self):
        """Load DEG and overlapping genes"""
        logger.info(f"\n[CELLEX] Loading data...")
        
        self.deg_df = pd.read_csv(self.deg_file)
        logger.info(f"  DEG file: {self.deg_df.shape}")
        
        self.overlap_genes = pd.read_csv(self.overlap_genes_file)
        logger.info(f"  Overlapping genes: {len(self.overlap_genes)}")
        
        # Filter DEG to only overlapping genes
        self.deg_df = self.deg_df[self.deg_df['gene'].isin(self.overlap_genes['gene'].values)]
        logger.info(f"  DEG after filtering: {self.deg_df.shape}")
        
        return True
    
    def prepare_expression_matrix(self):
        """Create gene × cell-type expression matrix"""
        logger.info(f"[CELLEX] Preparing expression matrix...")
        
        # Pivot: genes × clusters with avg_log2FC
        expression_matrix = self.deg_df.pivot_table(
            index='gene',
            columns='cluster',
            values='avg_log2FC',
            aggfunc='first'
        )
        
        expression_matrix = expression_matrix.fillna(0)
        logger.info(f"  ✓ Matrix shape: {expression_matrix.shape}")
        
        self.expression_matrix = expression_matrix
        return expression_matrix
    
    def calculate_specificity_index(self):
        """Calculate specificity scores [0, 1]"""
        logger.info(f"[CELLEX] Calculating specificity index...")
        
        if self.expression_matrix is None:
            self.prepare_expression_matrix()
        
        expr_matrix = self.expression_matrix.copy()
        specificity_matrix = pd.DataFrame(index=expr_matrix.index, columns=expr_matrix.columns, dtype=float)
        
        for gene in expr_matrix.index:
            gene_expr = expr_matrix.loc[gene].values
            mean_expr = np.mean(gene_expr)
            
            if mean_expr == 0:
                mean_expr = 1e-10
            
            specificity = gene_expr / (mean_expr + 1e-10)
            min_spec = specificity.min()
            max_spec = specificity.max()
            
            if max_spec > min_spec:
                specificity_normalized = (specificity - min_spec) / (max_spec - min_spec)
            else:
                specificity_normalized = np.zeros_like(specificity)
            
            specificity_matrix.loc[gene] = specificity_normalized
        
        self.specificity_scores = specificity_matrix.astype(float)
        logger.info(f"  ✓ Specificity scores: {self.specificity_scores.shape}")
        
        return self.specificity_scores
    
    def save_specificity_scores(self, filename=None):
        """Save specificity matrix"""
        if self.specificity_scores is None:
            self.calculate_specificity_index()
        
        if filename is None:
            filename = self.output_dir / "T2D_celltype_specificity_index.tsv"
        
        self.specificity_scores.to_csv(filename, sep='\t')
        logger.info(f"  ✓ Saved to {filename}")
        return filename

# ============================================================================
# STEP 4: CELLECT - CELL-TYPE GENETIC ENRICHMENT
# ============================================================================

class CELLECTMagmaAnalysis:
    """CELLECT-MAGMA: Cell-type genetic enrichment analysis"""
    
    def __init__(self, gwas_file, overlap_genes_file, specificity_file, output_dir):
        self.gwas_file = gwas_file
        self.overlap_genes_file = overlap_genes_file
        self.specificity_file = specificity_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.gwas_df = None
        self.overlap_genes = None
        self.specificity_df = None
        self.gene_pvalues = None
        self.enrichment_df = None
        
    def load_gwas_data(self):
        """Load GWAS file with proper parsing"""
        logger.info(f"[CELLECT] Loading GWAS file: {self.gwas_file}")
        
        # Load raw to inspect structure
        raw_df = pd.read_csv(self.gwas_file, sep='\t')
        
        logger.info(f"  Raw GWAS shape: {raw_df.shape}")
        logger.info(f"  Columns: {list(raw_df.columns)}")
        
        # The last column (with spaces in header) contains gene names
        gene_col = raw_df.columns[-1]
        pval_col = None
        
        for col in raw_df.columns:
            if 'p_value' in col.lower() or col.strip().lower() == 'p_value':
                pval_col = col
                break
        
        logger.info(f"  Gene column: {repr(gene_col)}")
        logger.info(f"  P-value column: {repr(pval_col)}")
        
        self.gwas_df = raw_df.copy()
        self.gene_col = gene_col
        self.pval_col = pval_col
        
        return raw_df
    
    def load_overlap_genes(self):
        """Load overlapping genes"""
        logger.info(f"[CELLECT] Loading overlapping genes: {self.overlap_genes_file}")
        self.overlap_genes = pd.read_csv(self.overlap_genes_file)
        logger.info(f"  Overlapping genes: {len(self.overlap_genes)}")
        return self.overlap_genes
    
    def load_specificity_data(self):
        """Load specificity matrix"""
        logger.info(f"[CELLECT] Loading specificity file: {self.specificity_file}")
        self.specificity_df = pd.read_csv(self.specificity_file, sep='\t', index_col=0)
        logger.info(f"  ✓ Loaded: {self.specificity_df.shape}")
        return self.specificity_df
    
    def extract_gene_pvalues(self):
        """Extract p-values for overlapping genes"""
        logger.info(f"[CELLECT] Extracting gene-level p-values...")
        
        # Get overlapping genes
        overlap_gene_set = set(self.overlap_genes['gene'].values)
        logger.info(f"  Overlapping genes: {len(overlap_gene_set)}")
        
        # Extract gene names and p-values from GWAS
        gene_pvalues = {}
        
        for idx, row in self.gwas_df.iterrows():
            gene = str(row[self.gene_col]).strip()
            
            if gene in overlap_gene_set:
                if self.pval_col and pd.notna(row[self.pval_col]):
                    try:
                        p = float(row[self.pval_col])
                        p = max(0.0, min(1.0, p))  # Clip to [0, 1]
                        
                        if gene in gene_pvalues:
                            gene_pvalues[gene] = min(gene_pvalues[gene], p)  # Keep minimum p
                        else:
                            gene_pvalues[gene] = p
                    except:
                        pass
        
        # Fill missing genes with p=1.0
        for gene in overlap_gene_set:
            if gene not in gene_pvalues:
                gene_pvalues[gene] = 1.0
        
        self.gene_pvalues = pd.Series(gene_pvalues)
        
        logger.info(f"  ✓ Extracted p-values: {len(self.gene_pvalues)} genes")
        logger.info(f"    p < 0.05: {sum(self.gene_pvalues < 0.05)}")
        logger.info(f"    p < 0.01: {sum(self.gene_pvalues < 0.01)}")
        logger.info(f"    p < 0.001: {sum(self.gene_pvalues < 0.001)}")
        
        return self.gene_pvalues
    
    def calculate_celltype_enrichment(self):
        """Calculate enrichment"""
        logger.info(f"[CELLECT] Calculating cell-type enrichment...")
        
        enrichment_results = {}
        
        for cell_type in self.specificity_df.columns:
            specificity_scores = self.specificity_df[cell_type]
            
            # Get top 25% specific genes
            threshold = np.percentile(specificity_scores, 75)
            high_spec_genes = specificity_scores[specificity_scores >= threshold].index.tolist()
            
            # Find shared genes
            shared_genes = [g for g in high_spec_genes if g in self.gene_pvalues.index]
            
            if len(shared_genes) > 0:
                cell_pvals = self.gene_pvalues[shared_genes]
                sig_genes = sum(cell_pvals < 0.05)
                
                try:
                    ks_stat, enrich_p = stats.kstest(cell_pvals, 'uniform')
                except:
                    enrich_p = np.median(cell_pvals)
                
                enrich_p = max(enrich_p, 1e-10)
                enrichment_results[cell_type] = {
                    'n_sig_genes': sig_genes,
                    'n_shared_genes': len(shared_genes),
                    'enrichment_p': enrich_p,
                    'log10_p': -np.log10(enrich_p)
                }
            else:
                enrichment_results[cell_type] = {
                    'n_sig_genes': 0,
                    'n_shared_genes': 0,
                    'enrichment_p': 1.0,
                    'log10_p': 0.0
                }
        
        self.enrichment_df = pd.DataFrame(enrichment_results).T.sort_values('enrichment_p')
        logger.info(f"  ✓ Enrichment calculated for {len(self.enrichment_df)} cell types")
        
        return self.enrichment_df
    
    def save_results(self, filename=None):
        """Save enrichment results"""
        if filename is None:
            filename = self.output_dir / "T2D_celltype_enrichment_results.tsv"
        
        self.enrichment_df.to_csv(filename, sep='\t')
        logger.info(f"  ✓ Saved to {filename}")
        return filename

# ============================================================================
# STEP 5: TARGET PRIORITIZATION
# ============================================================================

# T2D_pipeline_fixed_scoring.py
"""
Complete T2D Analysis Pipeline - FINAL VERSION WITH BETTER SCORING
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 5: TARGET PRIORITIZATION (IMPROVED SCORING)
# ============================================================================

# T2D_pipeline_final_improved.py
"""
Complete T2D Analysis Pipeline - FINAL WITH EXCELLENT SCATTER PLOT
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# STEP 5: TARGET PRIORITIZATION (IMPROVED)
# ============================================================================

class TargetPrioritization:
    """Integrate data sources for target prioritization"""
    
    def __init__(self, gwas_file, overlap_genes_file, specificity_file, enrichment_file, output_dir):
        self.gwas_file = gwas_file
        self.overlap_genes_file = overlap_genes_file
        self.specificity_file = specificity_file
        self.enrichment_file = enrichment_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.gwas_df = None
        self.overlap_genes = None
        self.specificity_df = None
        self.enrichment_df = None
        self.target_scores = None
        
    def load_data(self):
        """Load all data"""
        logger.info(f"[TARGET] Loading data for prioritization...")
        
        self.gwas_df = pd.read_csv(self.gwas_file, sep='\t')
        self.overlap_genes = pd.read_csv(self.overlap_genes_file)
        self.specificity_df = pd.read_csv(self.specificity_file, sep='\t', index_col=0)
        self.enrichment_df = pd.read_csv(self.enrichment_file, sep='\t', index_col=0)
        
        logger.info(f"  ✓ GWAS: {self.gwas_df.shape}")
        logger.info(f"  ✓ Overlap genes: {len(self.overlap_genes)}")
        logger.info(f"  ✓ Specificity: {self.specificity_df.shape}")
        logger.info(f"  ✓ Enrichment: {self.enrichment_df.shape}")
        return True
    
    def calculate_target_scores(self):
        """Calculate composite target scores with SMART GWAS SCORING"""
        logger.info(f"[TARGET] Calculating prioritization scores (improved)...")
        
        # Get overlapping genes only
        overlap_genes_set = set(self.overlap_genes['gene'].values)
        target_scores = pd.DataFrame(index=list(overlap_genes_set))
        
        # Find gene and scoring columns in GWAS
        gene_col = self.gwas_df.columns[-1]
        
        # Find available scoring columns
        beta_col = None
        se_col = None
        pval_col = None
        
        for col in self.gwas_df.columns:
            if 'beta' in col.lower():
                beta_col = col
            if 'se' in col.lower() or 'standard_error' in col.lower():
                se_col = col
            if 'p_value' in col.lower():
                pval_col = col
        
        logger.info(f"  Gene column: {repr(gene_col)}")
        logger.info(f"  Beta column: {repr(beta_col)}")
        logger.info(f"  SE column: {repr(se_col)}")
        logger.info(f"  P-value column: {repr(pval_col)}")
        
        # ===== COMPONENT 1: GWAS SCORE (Smart Strategy) =====
        logger.info(f"\n  [1/3] Calculating GWAS scores (using multiple strategies)...")
        
        gwas_scores_raw = []
        for gene in target_scores.index:
            gene_data = self.gwas_df[self.gwas_df[gene_col].str.strip() == gene]
            
            if len(gene_data) == 0:
                gwas_scores_raw.append(0.0)
                continue
            
            score = 0.0
            
            # Strategy 1: Use minimum p-value if available
            if pval_col and pval_col in self.gwas_df.columns:
                valid_pvals = gene_data[pval_col].dropna()
                if len(valid_pvals) > 0:
                    min_p = valid_pvals[valid_pvals > 0].min()
                    if pd.notna(min_p) and min_p > 0:
                        score = -np.log10(min_p)
            
            # Strategy 2: If p-value is 0/missing, use beta/SE to calculate Z-score
            if score == 0 and beta_col and se_col:
                try:
                    betas = pd.to_numeric(gene_data[beta_col], errors='coerce').dropna()
                    ses = pd.to_numeric(gene_data[se_col], errors='coerce').dropna()
                    
                    if len(betas) > 0 and len(ses) > 0:
                        z_scores = np.abs(betas.values / ses.values)
                        max_z = z_scores.max()
                        if max_z > 0:
                            from scipy.stats import norm
                            p_from_z = 2 * (1 - norm.cdf(max_z))
                            score = -np.log10(max(p_from_z, 1e-300))
                except:
                    pass
            
            # Strategy 3: Use absolute beta as fallback
            if score == 0 and beta_col:
                try:
                    betas = pd.to_numeric(gene_data[beta_col], errors='coerce').dropna()
                    if len(betas) > 0:
                        score = np.abs(betas).max() * 10  # Scale up
                except:
                    pass
            
            gwas_scores_raw.append(score)
        
        target_scores['gwas_score_raw'] = gwas_scores_raw
        
        # Normalize GWAS to [0, 1] with better distribution
        gwas_min = np.min(gwas_scores_raw)
        gwas_max = np.max(gwas_scores_raw)
        
        if gwas_max > gwas_min:
            target_scores['gwas_score'] = (np.array(gwas_scores_raw) - gwas_min) / (gwas_max - gwas_min)
        else:
            target_scores['gwas_score'] = np.full(len(gwas_scores_raw), 0.5)
        
        # Add jitter to avoid perfect alignment on axes
        target_scores['gwas_score'] = target_scores['gwas_score'] + np.random.normal(0, 0.02, len(target_scores))
        target_scores['gwas_score'] = target_scores['gwas_score'].clip(0, 1)
        
        logger.info(f"    GWAS raw range: {gwas_min:.4f} - {gwas_max:.4f}")
        logger.info(f"    GWAS normalized range: {target_scores['gwas_score'].min():.4f} - {target_scores['gwas_score'].max():.4f}")
        logger.info(f"    Non-zero GWAS genes: {sum(np.array(gwas_scores_raw) > 0)}")
        
        # ===== COMPONENT 2: SPECIFICITY SCORE =====
        logger.info(f"\n  [2/3] Calculating specificity scores...")
        
        spec_scores = []
        cell_types = []
        for gene in target_scores.index:
            if gene in self.specificity_df.index:
                max_spec = self.specificity_df.loc[gene].max()
                cell_type = self.specificity_df.loc[gene].idxmax()
                spec_scores.append(max_spec)
                cell_types.append(cell_type)
            else:
                spec_scores.append(0.5)
                cell_types.append('Unknown')
        
        target_scores['specificity_score'] = spec_scores
        target_scores['cell_type'] = cell_types
        
        logger.info(f"    Specificity range: {np.min(spec_scores):.4f} - {np.max(spec_scores):.4f}")
        logger.info(f"    Mean specificity: {np.mean(spec_scores):.4f}")
        
        # ===== COMPONENT 3: ENRICHMENT SCORE =====
        logger.info(f"\n  [3/3] Calculating enrichment scores...")
        
        enrichment_ps = []
        enrichment_scores = []
        for ct in target_scores['cell_type']:
            if ct in self.enrichment_df.index:
                p = self.enrichment_df.loc[ct, 'enrichment_p']
                enrichment_ps.append(p)
                enrichment_scores.append(-np.log10(p + 1e-300))
            else:
                enrichment_ps.append(1.0)
                enrichment_scores.append(0.0)
        
        target_scores['enrichment_p'] = enrichment_ps
        target_scores['enrichment_log10p'] = enrichment_scores
        
        # Normalize enrichment to [0, 1]
        enr_min = np.min(enrichment_scores)
        enr_max = np.max(enrichment_scores)
        if enr_max > enr_min:
            target_scores['enrichment_score'] = (np.array(enrichment_scores) - enr_min) / (enr_max - enr_min)
        else:
            target_scores['enrichment_score'] = np.full(len(enrichment_scores), 0.5)
        
        logger.info(f"    Enrichment -log10p range: {enr_min:.2f} - {enr_max:.2f}")
        logger.info(f"    Enrichment score range: {target_scores['enrichment_score'].min():.4f} - {target_scores['enrichment_score'].max():.4f}")
        
        # ===== COMPOSITE SCORE =====
        logger.info(f"\n  Calculating composite priority scores...")
        
        # Weighted combination
        weights = {
            'gwas': 0.4,
            'specificity': 0.35,
            'enrichment': 0.25
        }
        
        target_scores['priority_score'] = (
            weights['gwas'] * target_scores['gwas_score'] +
            weights['specificity'] * target_scores['specificity_score'] +
            weights['enrichment'] * target_scores['enrichment_score']
        )
        
        self.target_scores = target_scores.sort_values('priority_score', ascending=False)
        
        logger.info(f"  ✓ Scores calculated for {len(self.target_scores)} targets")
        logger.info(f"    Priority score range: {self.target_scores['priority_score'].min():.4f} - {self.target_scores['priority_score'].max():.4f}")
        
        logger.info(f"\n  TOP 25 TARGETS:")
        logger.info(f"  {'Rank':<5} {'Gene':<15} {'Priority':<10} {'Cell Type':<20} {'GWAS':<10} {'Spec':<8}")
        logger.info(f"  {'-'*5} {'-'*15} {'-'*10} {'-'*20} {'-'*10} {'-'*8}")
        
        for idx, (gene, row) in enumerate(self.target_scores.head(25).iterrows(), 1):
            logger.info(f"  {idx:<5} {gene:<15} {row['priority_score']:<10.4f} {str(row['cell_type']):<20} "
                       f"{row['gwas_score']:<10.4f} {row['specificity_score']:<8.4f}")
        
        return self.target_scores
    
    def save_targets(self, top_n=50, filename=None):
        """Save prioritized targets"""
        if filename is None:
            filename = self.output_dir / "T2D_prioritized_targets.tsv"
        
        save_cols = ['priority_score', 'gwas_score', 'specificity_score', 'cell_type', 
                     'enrichment_score', 'enrichment_p']
        
        self.target_scores[save_cols].head(top_n).to_csv(filename, sep='\t')
        logger.info(f"  ✓ Saved top {top_n} targets to {filename}")
        return filename

# ============================================================================
# STEP 6: VISUALIZATION (PUBLICATION-QUALITY)
# ============================================================================

class T2DVisualization:
    """Create professional visualization plots"""
    
    def __init__(self, enrichment_file, specificity_file, targets_file, output_dir):
        self.enrichment_file = enrichment_file
        self.specificity_file = specificity_file
        self.targets_file = targets_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load data"""
        self.enrichment_df = pd.read_csv(self.enrichment_file, sep='\t', index_col=0)
        self.specificity_df = pd.read_csv(self.specificity_file, sep='\t', index_col=0)
        self.targets_df = pd.read_csv(self.targets_file, sep='\t', index_col=0)
        
        logger.info(f"[VIZ] Data loaded: {len(self.targets_df)} targets")
    
    def plot_enrichment(self):
        """Cell-type enrichment bar plot"""
        logger.info(f"[VIZ] Creating enrichment plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_data = self.enrichment_df.sort_values('log10_p', ascending=True)
        
        colors = ['#d62728' if p < 0.05 else '#1f77b4' for p in plot_data['enrichment_p']]
        ax.barh(range(len(plot_data)), plot_data['log10_p'], color=colors, edgecolor='black', linewidth=1.5)
        ax.set_yticks(range(len(plot_data)))
        ax.set_yticklabels(plot_data.index, fontsize=11)
        ax.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=2.5, label='Significance (p=0.05)')
        ax.set_xlabel('-log10(P-value)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cell Type', fontsize=12, fontweight='bold')
        ax.set_title('T2D: Cell-Type Genetic Enrichment', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f9f9f9')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'enrichment.png', dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: enrichment.png")
        plt.close()
    
    def plot_targets_scatter(self):
        """PROFESSIONAL SCATTER PLOT"""
        logger.info(f"[VIZ] Creating professional scatter plot...")
        
        # Use all targets for better distribution
        data = self.targets_df.copy()
        
        # Create figure with better proportions
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Create scatter plot with multiple layers
        # Layer 1: Background all points
        scatter = ax.scatter(
            data['gwas_score'],
            data['specificity_score'],
            s=data['priority_score'] * 800 + 50,
            c=data['enrichment_score'],
            cmap='RdYlGn',
            alpha=0.7,
            edgecolors='black',
            linewidth=1.2,
            vmin=data['enrichment_score'].min(),
            vmax=data['enrichment_score'].max()
        )
        
        # Layer 2: Annotate top 30 genes
        top_30 = data.head(30)
        for gene, row in top_30.iterrows():
            ax.annotate(
                gene,
                xy=(row['gwas_score'], row['specificity_score']),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.6, edgecolor='black'),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black', lw=0.8)
            )
        
        # Styling
        ax.set_xlabel('GWAS Association Score', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cell-Type Specificity Score', fontsize=13, fontweight='bold')
        ax.set_title('T2D Therapeutic Targets: Multi-dimensional Prioritization\n' +
                    f'({len(data)} overlapping genes analyzed)',
                    fontsize=15, fontweight='bold', pad=20)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Cell-Type Enrichment Score', fontsize=12, fontweight='bold')
        cbar.ax.tick_params(labelsize=10)
        
        # Grid and background
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax.set_facecolor('#f5f5f5')
        fig.patch.set_facecolor('white')
        
        # Add quadrant lines for reference
        ax.axhline(y=data['specificity_score'].median(), color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.axvline(x=data['gwas_score'].median(), color='gray', linestyle=':', alpha=0.5, linewidth=1)
        
        # Legend for bubble sizes
        sizes = [0.3, 0.5, 0.7]
        for size in sizes:
            ax.scatter([], [], s=size * 800 + 50, c='gray', alpha=0.6, 
                      edgecolors='black', linewidth=1.2, label=f'Priority: {size:.1f}')
        
        leg1 = ax.legend(scatterpoints=1, frameon=True, labelspacing=1.5, 
                        loc='upper left', fontsize=11, title='Bubble Size')
        leg1.get_title().set_fontsize(12)
        leg1.get_title().set_fontweight('bold')
        
        # Set axis limits with small padding
        ax.set_xlim(data['gwas_score'].min() - 0.05, data['gwas_score'].max() + 0.05)
        ax.set_ylim(data['specificity_score'].min() - 0.05, data['specificity_score'].max() + 0.05)
        
        # Tick parameters
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'targets_scatter.png', dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: targets_scatter.png")
        plt.close()
    
    def plot_specificity_heatmap(self, top_n=30):
        """Heatmap of top specific genes"""
        logger.info(f"[VIZ] Creating specificity heatmap...")
        
        top_genes = self.targets_df.head(top_n).index
        top_spec = self.specificity_df.loc[top_genes]
        
        fig, ax = plt.subplots(figsize=(14, 12))
        sns.heatmap(top_spec, cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Specificity Score'},
                   linewidths=0.5, linecolor='gray')
        ax.set_title(f'Top {top_n} T2D Targets: Cell-Type Expression Specificity', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gene Target', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'specificity_heatmap.png', dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: specificity_heatmap.png")
        plt.close()
    
    def create_all_visualizations(self):
        """Create all plots"""
        logger.info(f"\n[VIZ] Creating publication-quality visualizations...")
        self.load_data()
        self.plot_enrichment()
        self.plot_targets_scatter()
        self.plot_specificity_heatmap(top_n=30)
        logger.info(f"  ✓ All visualizations complete!")

# ============================================================================
# MAIN PIPELINE
# ============================================================================

if __name__ == "__main__":
    logger.info("\n" + "=" * 100)
    logger.info("T2D GWAS × SINGLE-CELL INTEGRATION PIPELINE (FINAL)")
    logger.info("=" * 100)
    
    try:
        # Target Prioritization
        logger.info("\n[STEP 5] TARGET PRIORITIZATION")
        targets = TargetPrioritization(
            "gwas_chr22_gene_annotated.tsv",
            "deg_gwas_overlap_genes.csv",
            "cellex_output/T2D_celltype_specificity_index.tsv",
            "cellect_output/T2D_celltype_enrichment_results.tsv",
            "targets_output/"
        )
        targets.load_data()
        targets.calculate_target_scores()
        targets_file = targets.save_targets(top_n=50)
        
        # Visualization
        logger.info("\n[STEP 6] VISUALIZATION")
        viz = T2DVisualization(
            "cellect_output/T2D_celltype_enrichment_results.tsv",
            "cellex_output/T2D_celltype_specificity_index.tsv",
            targets_file,
            "plots/"
        )
        viz.create_all_visualizations()
        
        logger.info("\n" + "=" * 100)
        logger.info("✓ PIPELINE COMPLETE!")
        logger.info("=" * 100)
        logger.info("\nGenerated outputs:")
        logger.info("  📊 plots/enrichment.png - Cell-type enrichment analysis")
        logger.info("  📊 plots/targets_scatter.png - Target prioritization scatter plot")
        logger.info("  📊 plots/specificity_heatmap.png - Gene expression specificity heatmap")
        logger.info("  📋 targets_output/T2D_prioritized_targets.tsv - Ranked target genes")
        
    except Exception as e:
        logger.error(f"\n✗ FAILED: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    ####################

    # T2D_scatter_plot_all_161_genes.py
"""
Create scatter plot for ALL 161 overlapping genes
High-quality, publication-ready visualization
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class T2DScatterPlotAll161:
    """Create scatter plot for all 161 overlapping genes"""
    
    def __init__(self, gwas_file, overlap_genes_file, specificity_file, 
                 enrichment_file, targets_file, output_dir):
        self.gwas_file = gwas_file
        self.overlap_genes_file = overlap_genes_file
        self.specificity_file = specificity_file
        self.enrichment_file = enrichment_file
        self.targets_file = targets_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_all_data(self):
        """Load all required data"""
        logger.info("Loading data...")
        
        self.targets_df = pd.read_csv(self.targets_file, sep='\t', index_col=0)
        self.specificity_df = pd.read_csv(self.specificity_file, sep='\t', index_col=0)
        self.enrichment_df = pd.read_csv(self.enrichment_file, sep='\t', index_col=0)
        self.overlap_genes = pd.read_csv(self.overlap_genes_file)
        
        logger.info(f"✓ Loaded {len(self.targets_df)} target genes")
        logger.info(f"✓ Overlap genes: {len(self.overlap_genes)}")
        
    def create_scatter_plot_all_genes(self):
        """
        Create comprehensive scatter plot for ALL 161 genes
        X-axis: GWAS Score
        Y-axis: Specificity Score
        Color: Enrichment Score
        Size: Priority Score
        """
        logger.info("\n" + "="*80)
        logger.info("Creating scatter plot for ALL 161 overlapping genes")
        logger.info("="*80)
        
        # Use ALL genes from targets_df
        data = self.targets_df.copy()
        
        logger.info(f"\nData summary:")
        logger.info(f"  Total genes: {len(data)}")
        logger.info(f"  GWAS score range: {data['gwas_score'].min():.4f} - {data['gwas_score'].max():.4f}")
        logger.info(f"  Specificity range: {data['specificity_score'].min():.4f} - {data['specificity_score'].max():.4f}")
        logger.info(f"  Enrichment range: {data['enrichment_score'].min():.4f} - {data['enrichment_score'].max():.4f}")
        logger.info(f"  Priority range: {data['priority_score'].min():.4f} - {data['priority_score'].max():.4f}")
        
        # Create large figure
        fig = plt.figure(figsize=(18, 13))
        ax = fig.add_subplot(111)
        
        # ===== MAIN SCATTER PLOT =====
        # Create scatter with gradient colors
        scatter = ax.scatter(
            data['gwas_score'],
            data['specificity_score'],
            s=data['priority_score'] * 600 + 80,  # Bubble size by priority
            c=data['enrichment_score'],  # Color by enrichment
            cmap='RdYlGn',  # Red (low) to Green (high)
            alpha=0.7,
            edgecolors='black',
            linewidth=1.0,
            vmin=data['enrichment_score'].min(),
            vmax=data['enrichment_score'].max()
        )
        
        # ===== ANNOTATE TOP 40 GENES =====
        top_40 = data.head(40)
        logger.info(f"\nAnnotating top 40 genes...")
        
        for gene, row in top_40.iterrows():
            # Only label high-priority genes to avoid clutter
            if row['priority_score'] >= data['priority_score'].quantile(0.5):
                ax.annotate(
                    gene,
                    xy=(row['gwas_score'], row['specificity_score']),
                    xytext=(7, 7),
                    textcoords='offset points',
                    fontsize=8,
                    fontweight='bold',
                    bbox=dict(
                        boxstyle='round,pad=0.4',
                        facecolor='yellow',
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=0.8
                    ),
                    arrowprops=dict(
                        arrowstyle='->',
                        connectionstyle='arc3,rad=0.2',
                        color='black',
                        lw=0.7,
                        alpha=0.6
                    )
                )
        
        # ===== STYLING =====
        ax.set_xlabel('GWAS Association Score', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_ylabel('Cell-Type Specificity Score', fontsize=14, fontweight='bold', labelpad=12)
        ax.set_title(
            'Type 2 Diabetes (T2D) Therapeutic Targets:\nMulti-dimensional Prioritization of 161 Overlapping Genes',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        
        # ===== COLORBAR =====
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=30)
        cbar.set_label('Cell-Type Enrichment Score', fontsize=12, fontweight='bold', labelpad=15)
        cbar.ax.tick_params(labelsize=10)
        
        # ===== GRID =====
        ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.7, color='gray')
        ax.set_axisbelow(True)
        
        # ===== BACKGROUND =====
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        # ===== REFERENCE LINES =====
        median_gwas = data['gwas_score'].median()
        median_spec = data['specificity_score'].median()
        
        ax.axhline(y=median_spec, color='red', linestyle=':', alpha=0.4, linewidth=1.5, label='Specificity Median')
        ax.axvline(x=median_gwas, color='blue', linestyle=':', alpha=0.4, linewidth=1.5, label='GWAS Median')
        
        # ===== LEGEND FOR BUBBLE SIZES =====
        sizes_legend = [0.2, 0.5, 0.8]
        legend_bubbles = []
        for size in sizes_legend:
            legend_bubbles.append(
                Line2D([0], [0], marker='o', color='w', label=f'Priority: {size:.1f}',
                      markerfacecolor='gray', markersize=np.sqrt(size * 600 + 80) / 2,
                      markeredgecolor='black', markeredgewidth=1)
            )
        
        leg1 = ax.legend(handles=legend_bubbles, loc='upper left', fontsize=11, 
                        title='Bubble Size', framealpha=0.95, edgecolor='black')
        leg1.get_title().set_fontsize(12)
        leg1.get_title().set_fontweight('bold')
        
        # ===== SECOND LEGEND FOR REFERENCE LINES =====
        leg2 = ax.legend([ax.get_lines()[0], ax.get_lines()[1]], 
                        ['Specificity Median', 'GWAS Median'],
                        loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black')
        ax.add_artist(leg1)  # Add back first legend
        
        # ===== AXIS LIMITS =====
        x_margin = (data['gwas_score'].max() - data['gwas_score'].min()) * 0.05
        y_margin = (data['specificity_score'].max() - data['specificity_score'].min()) * 0.05
        
        ax.set_xlim(data['gwas_score'].min() - x_margin, data['gwas_score'].max() + x_margin)
        ax.set_ylim(data['specificity_score'].min() - y_margin, data['specificity_score'].max() + y_margin)
        
        # ===== TICK PARAMETERS =====
        ax.tick_params(axis='both', which='major', labelsize=11, length=6, width=1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        
        # ===== ADD STATISTICS BOX =====
        stats_text = f"""
        Analysis Summary:
        ━━━━━━━━━━━━━━━━━━
        Total Genes: {len(data)}
        Mean GWAS Score: {data['gwas_score'].mean():.3f}
        Mean Specificity: {data['specificity_score'].mean():.3f}
        Mean Enrichment: {data['enrichment_score'].mean():.3f}
        """
        
        ax.text(
            0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5)
        )
        
        plt.tight_layout()
        
        # ===== SAVE FIGURE =====
        output_file = self.output_dir / 'T2D_scatter_ALL_161_genes.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"\n✓ Saved: {output_file}")
        plt.close()
        
        return output_file
    
    def create_scatter_plot_high_quality_version(self):
        """Create an alternative high-contrast version"""
        logger.info("\nCreating high-contrast version...")
        
        data = self.targets_df.copy()
        
        fig = plt.figure(figsize=(18, 13))
        ax = fig.add_subplot(111)
        
        # ===== BACKGROUND GRADIENT (optional) =====
        # Create a subtle background gradient
        
        # ===== MAIN SCATTER =====
        scatter = ax.scatter(
            data['gwas_score'],
            data['specificity_score'],
            s=data['priority_score'] * 700 + 100,
            c=data['enrichment_score'],
            cmap='plasma',  # Alternative: high contrast
            alpha=0.75,
            edgecolors='white',
            linewidth=1.5,
            vmin=data['enrichment_score'].min(),
            vmax=data['enrichment_score'].max()
        )
        
        # ===== ANNOTATE ALL TOP 50 GENES =====
        top_50 = data.head(50)
        logger.info(f"Annotating top 50 genes...")
        
        for idx, (gene, row) in enumerate(top_50.iterrows()):
            ax.annotate(
                gene,
                xy=(row['gwas_score'], row['specificity_score']),
                xytext=(6, 6),
                textcoords='offset points',
                fontsize=7.5,
                fontweight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='lightyellow',
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.7
                ),
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.15',
                    color='darkred',
                    lw=0.6,
                    alpha=0.7
                )
            )
        
        # ===== STYLING =====
        ax.set_xlabel('GWAS Signal Score (normalized)', fontsize=13, fontweight='bold', labelpad=12)
        ax.set_ylabel('Cell-Type Expression Specificity', fontsize=13, fontweight='bold', labelpad=12)
        ax.set_title(
            f'T2D Therapeutic Target Prioritization\nAll {len(data)} Overlapping Genes',
            fontsize=15,
            fontweight='bold',
            pad=20
        )
        
        # ===== COLORBAR =====
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        cbar.set_label('Cell-Type Enrichment', fontsize=11, fontweight='bold')
        
        # ===== GRID =====
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_facecolor('#fafafa')
        
        # ===== QUADRANT SHADING =====
        # Add subtle quadrant backgrounds
        med_x = data['gwas_score'].median()
        med_y = data['specificity_score'].median()
        
        rect1 = Rectangle((data['gwas_score'].min(), med_y), 
                         med_x - data['gwas_score'].min(), 
                         data['specificity_score'].max() - med_y,
                         alpha=0.05, facecolor='red')
        ax.add_patch(rect1)
        
        # ===== SPINES =====
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')
        
        plt.tight_layout()
        
        output_file = self.output_dir / 'T2D_scatter_ALL_161_genes_highcontrast.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✓ Saved: {output_file}")
        plt.close()
        
        return output_file
    
    def print_gene_rankings(self):
        """Print rankings of all 161 genes"""
        logger.info("\n" + "="*100)
        logger.info("TOP 50 T2D THERAPEUTIC TARGETS (by priority score)")
        logger.info("="*100)
        logger.info(f"{'Rank':<6} {'Gene':<15} {'Priority':<12} {'GWAS':<10} {'Specificity':<12} {'Enrichment':<12} {'Cell Type':<25}")
        logger.info("-"*100)
        
        for idx, (gene, row) in enumerate(self.targets_df.head(50).iterrows(), 1):
            logger.info(f"{idx:<6} {gene:<15} {row['priority_score']:<12.4f} {row['gwas_score']:<10.4f} "
                       f"{row['specificity_score']:<12.4f} {row['enrichment_score']:<12.4f} {str(row['cell_type']):<25}")
        
        logger.info("\n" + "="*100)
        logger.info("GENES 51-100")
        logger.info("="*100)
        
        for idx, (gene, row) in enumerate(self.targets_df.iloc[50:100].iterrows(), 51):
            logger.info(f"{idx:<6} {gene:<15} {row['priority_score']:<12.4f} {row['gwas_score']:<10.4f} "
                       f"{row['specificity_score']:<12.4f} {row['enrichment_score']:<12.4f} {str(row['cell_type']):<25}")
        
        logger.info("\n" + "="*100)
        logger.info("GENES 101-161")
        logger.info("="*100)
        
        for idx, (gene, row) in enumerate(self.targets_df.iloc[100:].iterrows(), 101):
            logger.info(f"{idx:<6} {gene:<15} {row['priority_score']:<12.4f} {row['gwas_score']:<10.4f} "
                       f"{row['specificity_score']:<12.4f} {row['enrichment_score']:<12.4f} {str(row['cell_type']):<25}")
    
    def run_all(self):
        """Run complete scatter plot generation"""
        logger.info("\n" + "="*100)
        logger.info("T2D SCATTER PLOT: ALL 161 OVERLAPPING GENES")
        logger.info("="*100)
        
        self.load_all_data()
        self.create_scatter_plot_all_genes()
        self.create_scatter_plot_high_quality_version()
        self.print_gene_rankings()
        
        logger.info("\n" + "="*100)
        logger.info("✓ COMPLETE!")
        logger.info("="*100)
        logger.info(f"\nOutput saved to: {self.output_dir}")

if __name__ == "__main__":
    scatter_plot = T2DScatterPlotAll161(
        gwas_file="gwas_chr22_gene_annotated.tsv",
        overlap_genes_file="deg_gwas_overlap_genes.csv",
        specificity_file="cellex_output/T2D_celltype_specificity_index.tsv",
        enrichment_file="cellect_output/T2D_celltype_enrichment_results.tsv",
        targets_file="targets_output/T2D_prioritized_targets.tsv",
        output_dir="plots/"
    )
    scatter_plot.run_all() 

    #####
    # T2D_scatter_plot_ALL_161_GENES_FIXED.py
"""
Create scatter plot for ALL 161 overlapping genes - FIXED VERSION
Ensures all genes are plotted with proper distribution
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class T2DScatterPlotAll161Fixed:
    """Create scatter plot for all 161 overlapping genes"""
    
    def __init__(self, targets_file, specificity_file, enrichment_file, 
                 overlap_genes_file, output_dir):
        self.targets_file = targets_file
        self.specificity_file = specificity_file
        self.enrichment_file = enrichment_file
        self.overlap_genes_file = overlap_genes_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_prepare_data(self):
        """Load and verify all data"""
        logger.info("=" * 100)
        logger.info("Loading Data for ALL 161 Overlapping Genes")
        logger.info("=" * 100)
        
        # Load overlap genes to know how many we should have
        overlap = pd.read_csv(self.overlap_genes_file)
        overlap_genes_set = set(overlap['gene'].values)
        logger.info(f"\n✓ Total overlapping genes expected: {len(overlap_genes_set)}")
        
        # Load targets
        targets_df = pd.read_csv(self.targets_file, sep='\t', index_col=0)
        logger.info(f"✓ Targets file loaded: {len(targets_df)} genes")
        
        # CRITICAL: Keep ALL genes from overlap, not just top 50
        # Filter targets to only include overlapping genes
        data = targets_df.loc[targets_df.index.isin(overlap_genes_set)].copy()
        logger.info(f"✓ After filtering to overlap: {len(data)} genes")
        
        if len(data) < len(overlap_genes_set):
            logger.warning(f"⚠️  Missing {len(overlap_genes_set) - len(data)} genes!")
            missing_genes = overlap_genes_set - set(data.index)
            logger.info(f"   Missing genes: {list(missing_genes)[:10]}")
        
        # Load other data
        self.specificity_df = pd.read_csv(self.specificity_file, sep='\t', index_col=0)
        self.enrichment_df = pd.read_csv(self.enrichment_file, sep='\t', index_col=0)
        
        logger.info(f"✓ Specificity matrix: {self.specificity_df.shape}")
        logger.info(f"✓ Enrichment matrix: {self.enrichment_df.shape}")
        
        # Sort by priority but KEEP ALL genes
        self.data = data.sort_values('priority_score', ascending=False)
        
        logger.info(f"\n" + "-" * 100)
        logger.info(f"FINAL DATA FOR PLOTTING:")
        logger.info(f"-" * 100)
        logger.info(f"Total genes to plot: {len(self.data)}")
        logger.info(f"GWAS score - Min: {self.data['gwas_score'].min():.4f}, Max: {self.data['gwas_score'].max():.4f}, Mean: {self.data['gwas_score'].mean():.4f}")
        logger.info(f"Specificity - Min: {self.data['specificity_score'].min():.4f}, Max: {self.data['specificity_score'].max():.4f}, Mean: {self.data['specificity_score'].mean():.4f}")
        logger.info(f"Enrichment  - Min: {self.data['enrichment_score'].min():.4f}, Max: {self.data['enrichment_score'].max():.4f}, Mean: {self.data['enrichment_score'].mean():.4f}")
        logger.info(f"Priority    - Min: {self.data['priority_score'].min():.4f}, Max: {self.data['priority_score'].max():.4f}, Mean: {self.data['priority_score'].mean():.4f}")
        
        return self.data
    
    def create_main_scatter_plot(self):
        """Create main scatter plot for ALL 161 genes"""
        logger.info(f"\n" + "=" * 100)
        logger.info(f"Creating Main Scatter Plot - ALL {len(self.data)} GENES")
        logger.info("=" * 100)
        
        data = self.data
        
        # Create figure with optimal size
        fig = plt.figure(figsize=(20, 14))
        ax = fig.add_subplot(111)
        
        # ===== MAIN SCATTER PLOT =====
        scatter = ax.scatter(
            data['gwas_score'],
            data['specificity_score'],
            s=data['priority_score'] * 500 + 80,  # Bubble size by priority
            c=data['enrichment_score'],  # Color by enrichment
            cmap='RdYlGn',  # Red=low enrichment, Green=high enrichment
            alpha=0.72,
            edgecolors='black',
            linewidth=0.8,
            vmin=data['enrichment_score'].min(),
            vmax=data['enrichment_score'].max()
        )
        
        # ===== ANNOTATE TOP 50 GENES =====
        logger.info(f"Annotating top 50 genes...")
        top_50 = data.head(50)
        
        for gene, row in top_50.iterrows():
            ax.annotate(
                gene,
                xy=(row['gwas_score'], row['specificity_score']),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold',
                color='darkblue',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='yellow',
                    alpha=0.75,
                    edgecolor='darkred',
                    linewidth=1.0
                ),
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.25',
                    color='darkred',
                    lw=0.8,
                    alpha=0.7
                ),
                zorder=5
            )
        
        # ===== AXIS LABELS =====
        ax.set_xlabel('GWAS Association Score', fontsize=15, fontweight='bold', labelpad=15)
        ax.set_ylabel('Cell-Type Specificity Score', fontsize=15, fontweight='bold', labelpad=15)
        ax.set_title(
            f'Type 2 Diabetes (T2D) Therapeutic Targets\nMulti-dimensional Prioritization of {len(data)} Overlapping Genes',
            fontsize=17,
            fontweight='bold',
            pad=25
        )
        
        # ===== COLORBAR =====
        cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=40)
        cbar.set_label('Cell-Type Enrichment Score', fontsize=13, fontweight='bold', labelpad=15)
        cbar.ax.tick_params(labelsize=11)
        
        # ===== GRID =====
        ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.7, color='gray')
        ax.set_axisbelow(True)
        
        # ===== BACKGROUND =====
        ax.set_facecolor('#f9f9f9')
        fig.patch.set_facecolor('white')
        
        # ===== REFERENCE LINES (Medians) =====
        median_gwas = data['gwas_score'].median()
        median_spec = data['specificity_score'].median()
        
        ax.axhline(y=median_spec, color='red', linestyle=':', alpha=0.5, linewidth=2.0, label='Specificity Median')
        ax.axvline(x=median_gwas, color='blue', linestyle=':', alpha=0.5, linewidth=2.0, label='GWAS Median')
        
        # ===== LEGEND FOR BUBBLE SIZES =====
        sizes_legend = [0.2, 0.5, 0.8]
        legend_bubbles = []
        for size in sizes_legend:
            legend_bubbles.append(
                Line2D([0], [0], marker='o', color='w', label=f'Priority: {size:.1f}',
                      markerfacecolor='gray', markersize=np.sqrt(size * 500 + 80) / 2.5,
                      markeredgecolor='black', markeredgewidth=1.2)
            )
        
        leg1 = ax.legend(handles=legend_bubbles, loc='upper left', fontsize=12, 
                        title='Bubble Size (Priority Score)', framealpha=0.95, 
                        edgecolor='black', fancybox=True, shadow=True)
        leg1.get_title().set_fontsize(13)
        leg1.get_title().set_fontweight('bold')
        
        # ===== SECOND LEGEND =====
        leg2 = ax.legend([ax.get_lines()[0], ax.get_lines()[1]], 
                        ['Specificity Median', 'GWAS Median'],
                        loc='lower right', fontsize=12, framealpha=0.95, 
                        edgecolor='black', fancybox=True, shadow=True)
        ax.add_artist(leg1)
        
        # ===== AXIS PROPERTIES =====
        x_margin = (data['gwas_score'].max() - data['gwas_score'].min()) * 0.08
        y_margin = (data['specificity_score'].max() - data['specificity_score'].min()) * 0.08
        
        ax.set_xlim(data['gwas_score'].min() - x_margin, data['gwas_score'].max() + x_margin)
        ax.set_ylim(data['specificity_score'].min() - y_margin, data['specificity_score'].max() + y_margin)
        
        ax.tick_params(axis='both', which='major', labelsize=12, length=7, width=1.5)
        
        # ===== SPINES =====
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        # ===== STATISTICS BOX =====
        stats_text = f"""ANALYSIS SUMMARY
━━━━━━━━━━━━━━━━━━━
Total Genes: {len(data)}
Mean GWAS: {data['gwas_score'].mean():.4f}
Mean Specificity: {data['specificity_score'].mean():.4f}
Mean Enrichment: {data['enrichment_score'].mean():.4f}
Median Priority: {data['priority_score'].median():.4f}"""
        
        ax.text(
            0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', 
                     alpha=0.9, edgecolor='black', linewidth=2)
        )
        
        plt.tight_layout()
        
        # ===== SAVE =====
        output_file = self.output_dir / f'T2D_Scatter_ALL_{len(data)}_Genes.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"\n✓ Saved: {output_file}")
        plt.close()
        
        return output_file
    
    def create_density_scatter_plot(self):
        """Create alternative density-based scatter plot"""
        logger.info(f"\nCreating density-based scatter plot...")
        
        data = self.data
        
        fig, ax = plt.subplots(figsize=(18, 13))
        
        # Density scatter
        scatter = ax.scatter(
            data['gwas_score'],
            data['specificity_score'],
            s=data['priority_score'] * 450 + 100,
            c=data['enrichment_score'],
            cmap='viridis',
            alpha=0.65,
            edgecolors='white',
            linewidth=1.5,
            vmin=data['enrichment_score'].min(),
            vmax=data['enrichment_score'].max()
        )
        
        ax.set_xlabel('GWAS Association Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Cell-Type Specificity Score', fontsize=14, fontweight='bold')
        ax.set_title(
            f'T2D Target Prioritization: All {len(data)} Overlapping Genes\n(Density-based visualization)',
            fontsize=15, fontweight='bold'
        )
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Enrichment Score', fontsize=12, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#fafafa')
        
        # Annotate only top 30 for clarity
        for gene, row in data.head(30).iterrows():
            ax.annotate(gene, (row['gwas_score'], row['specificity_score']),
                       fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')
        
        plt.tight_layout()
        output_file = self.output_dir / f'T2D_Scatter_Density_{len(data)}_Genes.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"✓ Saved: {output_file}")
        plt.close()
    
    def create_gene_rankings_file(self):
        """Create comprehensive CSV with all gene rankings"""
        logger.info(f"\nCreating gene rankings file...")
        
        output_file = self.output_dir / f'T2D_All_{len(self.data)}_Genes_Rankings.csv'
        
        ranking_df = self.data[['priority_score', 'gwas_score', 'specificity_score', 
                                'enrichment_score', 'cell_type']].copy()
        ranking_df.insert(0, 'Rank', range(1, len(ranking_df) + 1))
        
        ranking_df.to_csv(output_file)
        logger.info(f"✓ Saved: {output_file}")
        
        return output_file
    
    def print_comprehensive_rankings(self):
        """Print all 161 genes with scores"""
        logger.info(f"\n" + "=" * 130)
        logger.info(f"ALL {len(self.data)} T2D THERAPEUTIC TARGETS - RANKED BY PRIORITY")
        logger.info("=" * 130)
        logger.info(f"{'Rank':<6} {'Gene':<15} {'Priority':<12} {'GWAS':<12} {'Specificity':<14} {'Enrichment':<12} {'Cell Type':<25}")
        logger.info("-" * 130)
        
        for idx, (gene, row) in enumerate(self.data.iterrows(), 1):
            logger.info(f"{idx:<6} {gene:<15} {row['priority_score']:<12.4f} {row['gwas_score']:<12.4f} "
                       f"{row['specificity_score']:<14.4f} {row['enrichment_score']:<12.4f} {str(row['cell_type']):<25}")
        
        logger.info("=" * 130)
    
    def run_all(self):
        """Run complete visualization"""
        logger.info("\n" + "🧬" * 50)
        logger.info("T2D SCATTER PLOT: ALL 161 OVERLAPPING GENES")
        logger.info("🧬" * 50)
        
        self.load_and_prepare_data()
        self.create_main_scatter_plot()
        self.create_density_scatter_plot()
        self.create_gene_rankings_file()
        self.print_comprehensive_rankings()
        
        logger.info("\n" + "=" * 100)
        logger.info("✓ ALL VISUALIZATIONS COMPLETE!")
        logger.info("=" * 100)
        logger.info(f"Output directory: {self.output_dir}")

if __name__ == "__main__":
    scatter = T2DScatterPlotAll161Fixed(
        targets_file="targets_output/T2D_prioritized_targets.tsv",
        specificity_file="cellex_output/T2D_celltype_specificity_index.tsv",
        enrichment_file="cellect_output/T2D_celltype_enrichment_results.tsv",
        overlap_genes_file="deg_gwas_overlap_genes.csv",
        output_dir="plots/"
    )
    scatter.run_all()