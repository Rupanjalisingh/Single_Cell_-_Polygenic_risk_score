# T2D_visualizations_from_outputs.py
"""
Create comprehensive visualizations directly from CELLECT and CELLEX outputs
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class CELLECTCELLEXVisualizations:
    """Create visualizations from CELLECT and CELLEX outputs"""
    
    def __init__(self, specificity_file, enrichment_file, output_dir="visualizations/"):
        self.specificity_file = specificity_file
        self.enrichment_file = enrichment_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.specificity_df = None
        self.enrichment_df = None
        
    def load_data(self):
        """Load CELLEX and CELLECT outputs"""
        logger.info("=" * 100)
        logger.info("Loading CELLEX and CELLECT Data")
        logger.info("=" * 100)
        
        # Load specificity matrix from CELLEX
        self.specificity_df = pd.read_csv(self.specificity_file, sep='\t', index_col=0)
        logger.info(f"\n✓ CELLEX Specificity Matrix:")
        logger.info(f"  Shape: {self.specificity_df.shape}")
        logger.info(f"  Genes: {self.specificity_df.shape[0]}")
        logger.info(f"  Cell Types: {self.specificity_df.shape[1]}")
        logger.info(f"  Cell Types: {list(self.specificity_df.columns)}")
        
        # Load enrichment from CELLECT
        self.enrichment_df = pd.read_csv(self.enrichment_file, sep='\t', index_col=0)
        logger.info(f"\n✓ CELLECT Enrichment Results:")
        logger.info(f"  Shape: {self.enrichment_df.shape}")
        logger.info(f"  Columns: {list(self.enrichment_df.columns)}")
        
        return True
    
    # ========================================================================
    # VISUALIZATION 1: CELLECT ENRICHMENT BAR PLOT
    # ========================================================================
    
    def plot_cellect_enrichment(self):
        """Cell-type enrichment bar plot from CELLECT"""
        logger.info("\n[VIZ 1/8] Creating CELLECT enrichment plot...")
        
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # Sort by enrichment p-value
        plot_data = self.enrichment_df.sort_values('log10_p', ascending=True)
        
        # Color based on significance
        colors = ['#d62728' if p < 0.05 else '#1f77b4' for p in plot_data['enrichment_p']]
        
        # Create bar plot
        bars = ax.barh(range(len(plot_data)), plot_data['log10_p'], color=colors, 
                       edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(plot_data.iterrows()):
            ax.text(row['log10_p'] + 0.05, i, f"{row['log10_p']:.2f}", 
                   va='center', fontsize=9, fontweight='bold')
        
        ax.set_yticks(range(len(plot_data)))
        ax.set_yticklabels(plot_data.index, fontsize=11, fontweight='bold')
        ax.axvline(-np.log10(0.05), color='red', linestyle='--', linewidth=2.5, 
                  label='Significance threshold (p=0.05)')
        
        ax.set_xlabel('-log10(P-value)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cell Type', fontsize=13, fontweight='bold')
        ax.set_title('CELLECT-MAGMA: T2D Genetic Enrichment by Cell Type\n' + 
                    'Higher bars indicate stronger cell-type relevance to T2D',
                    fontsize=15, fontweight='bold', pad=20)
        
        ax.legend(fontsize=11, loc='lower right')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f9f9f9')
        
        # Add statistics
        sig_cells = sum(plot_data['enrichment_p'] < 0.05)
        stats_text = f"Significant Cell Types (p<0.05): {sig_cells}/{len(plot_data)}"
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, 
               fontsize=10, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '1_CELLECT_Enrichment.png', dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: 1_CELLECT_Enrichment.png")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 2: CELLEX SPECIFICITY HEATMAP (ALL GENES)
    # ========================================================================
    
    def plot_cellex_specificity_heatmap_top_genes(self, top_n=50):
        """Top genes by mean specificity - heatmap"""
        logger.info(f"\n[VIZ 2/8] Creating CELLEX specificity heatmap (top {top_n} genes)...")
        
        # Calculate mean specificity per gene
        gene_means = self.specificity_df.mean(axis=1).sort_values(ascending=False)
        top_genes = gene_means.head(top_n).index
        
        # Get heatmap data
        heatmap_data = self.specificity_df.loc[top_genes]
        
        fig, ax = plt.subplots(figsize=(14, 16))
        
        sns.heatmap(heatmap_data, cmap='RdYlGn', ax=ax, 
                   cbar_kws={'label': 'Specificity Score'},
                   linewidths=0.5, linecolor='gray')
        
        ax.set_title(f'CELLEX: Top {top_n} Genes by Mean Cell-Type Specificity',
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Gene', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'2_CELLEX_Specificity_Heatmap_Top{top_n}.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: 2_CELLEX_Specificity_Heatmap_Top{top_n}.png")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 3: CELLEX SPECIFICITY BY CELL TYPE
    # ========================================================================
    
    def plot_cellex_specificity_by_celltype(self):
        """Distribution of specificity scores for each cell type"""
        logger.info(f"\n[VIZ 3/8] Creating CELLEX specificity by cell type...")
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Prepare data for violin plot
        plot_data = []
        labels = []
        for cell_type in self.specificity_df.columns:
            plot_data.append(self.specificity_df[cell_type].values)
            labels.append(cell_type)
        
        # Create violin plot
        parts = ax.violinplot(plot_data, positions=range(len(labels)), 
                             showmeans=True, showmedians=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(plt.cm.Set3(i))
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Specificity Score', fontsize=12, fontweight='bold')
        ax.set_title('CELLEX: Distribution of Gene Specificity Scores by Cell Type',
                    fontsize=15, fontweight='bold', pad=20)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f9f9f9')
        
        # Add statistics
        for i, cell_type in enumerate(self.specificity_df.columns):
            mean_spec = self.specificity_df[cell_type].mean()
            ax.text(i, 1.05, f'{mean_spec:.3f}', ha='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '3_CELLEX_Specificity_Distribution.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: 3_CELLEX_Specificity_Distribution.png")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 4: CELLEX SPECIFICITY SUMMARY (BOX PLOT)
    # ========================================================================
    
    def plot_cellex_specificity_boxplot(self):
        """Box plot of specificity by cell type"""
        logger.info(f"\n[VIZ 4/8] Creating CELLEX specificity box plot...")
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Prepare data
        box_data = [self.specificity_df[col].values for col in self.specificity_df.columns]
        
        # Create box plot
        bp = ax.boxplot(box_data, labels=self.specificity_df.columns, patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.specificity_df.columns)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_xticklabels(self.specificity_df.columns, rotation=45, ha='right', fontsize=11)
        ax.set_ylabel('Specificity Score', fontsize=12, fontweight='bold')
        ax.set_title('CELLEX: Gene Specificity Statistics by Cell Type',
                    fontsize=15, fontweight='bold', pad=20)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f9f9f9')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '4_CELLEX_Specificity_Boxplot.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: 4_CELLEX_Specificity_Boxplot.png")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 5: HEATMAP OF CELL-TYPE SPECIFICITY PATTERNS
    # ========================================================================
    
    def plot_celltype_specificity_patterns(self):
        """Cell type similarity based on specificity patterns"""
        logger.info(f"\n[VIZ 5/8] Creating cell-type similarity patterns...")
        
        # Calculate correlation between cell types based on their specificity patterns
        celltype_corr = self.specificity_df.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(celltype_corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax, cbar_kws={'label': 'Correlation'},
                   linewidths=1, linecolor='gray')
        
        ax.set_title('CELLEX: Cell-Type Similarity\n' + 
                    '(Based on gene specificity patterns)',
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xlabel('Cell Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cell Type', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '5_CellType_Similarity_Patterns.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: 5_CellType_Similarity_Patterns.png")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 6: SCATTER - ENRICHMENT VS MEAN SPECIFICITY
    # ========================================================================
    
    def plot_enrichment_vs_specificity(self):
        """Scatter: CELLECT enrichment vs CELLEX mean specificity"""
        logger.info(f"\n[VIZ 6/8] Creating enrichment vs specificity scatter...")
        
        # Calculate mean specificity per cell type
        mean_specificity = self.specificity_df.mean()
        
        # Combine with enrichment data
        scatter_data = pd.DataFrame({
            'mean_specificity': mean_specificity,
            'enrichment_p': self.enrichment_df['enrichment_p'],
            'log10_p': self.enrichment_df['log10_p']
        })
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        # Color by significance
        colors = ['#d62728' if p < 0.05 else '#1f77b4' for p in scatter_data['enrichment_p']]
        
        scatter = ax.scatter(scatter_data['mean_specificity'], 
                            scatter_data['log10_p'],
                            s=300, c=colors, alpha=0.7, edgecolors='black', linewidth=2)
        
        # Annotate cell types
        for cell_type in scatter_data.index:
            ax.annotate(cell_type, 
                       (scatter_data.loc[cell_type, 'mean_specificity'],
                        scatter_data.loc[cell_type, 'log10_p']),
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.6))
        
        ax.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=2, 
                  label='Significance (p=0.05)')
        
        ax.set_xlabel('Mean Gene Specificity (CELLEX)', fontsize=12, fontweight='bold')
        ax.set_ylabel('CELLECT -log10(P-value)', fontsize=12, fontweight='bold')
        ax.set_title('Integration of CELLEX and CELLECT:\n' + 
                    'Cell-Type Specificity vs Genetic Enrichment',
                    fontsize=15, fontweight='bold', pad=20)
        
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#f9f9f9')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '6_Enrichment_vs_Specificity.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: 6_Enrichment_vs_Specificity.png")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 7: TOP CELL-TYPE SPECIFIC GENES PER CELL TYPE
    # ========================================================================
    
    def plot_top_genes_per_celltype(self, top_n=5):
        """Bar plot of top N specific genes for each cell type"""
        logger.info(f"\n[VIZ 7/8] Creating top {top_n} genes per cell type...")
        
        fig, axes = plt.subplots(4, 4, figsize=(18, 14))
        axes = axes.flatten()
        
        for idx, cell_type in enumerate(self.specificity_df.columns):
            ax = axes[idx]
            
            # Get top genes for this cell type
            top_genes = self.specificity_df[cell_type].nlargest(top_n)
            
            # Create bar plot
            ax.barh(range(len(top_genes)), top_genes.values, color='steelblue', edgecolor='black')
            ax.set_yticks(range(len(top_genes)))
            ax.set_yticklabels(top_genes.index, fontsize=9)
            ax.set_xlabel('Specificity Score', fontsize=9, fontweight='bold')
            ax.set_title(cell_type, fontsize=10, fontweight='bold', color='darkblue')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(top_genes.values):
                ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontsize=8)
        
        # Hide extra subplots
        for idx in range(len(self.specificity_df.columns), len(axes)):
            axes[idx].axis('off')
        
        fig.suptitle(f'CELLEX: Top {top_n} Most Specific Genes per Cell Type',
                    fontsize=17, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'7_Top{top_n}_Genes_Per_CellType.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: 7_Top{top_n}_Genes_Per_CellType.png")
        plt.close()
    
    # ========================================================================
    # VISUALIZATION 8: SUMMARY STATISTICS TABLE
    # ========================================================================
    
    def create_summary_statistics_figure(self):
        """Create summary statistics as figure"""
        logger.info(f"\n[VIZ 8/8] Creating summary statistics figure...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # ===== LEFT PANEL: CELLEX Statistics =====
        cellex_stats = []
        for cell_type in self.specificity_df.columns:
            specs = self.specificity_df[cell_type]
            cellex_stats.append({
                'Cell Type': cell_type,
                'Mean': f"{specs.mean():.4f}",
                'Median': f"{specs.median():.4f}",
                'Std': f"{specs.std():.4f}",
                'Min': f"{specs.min():.4f}",
                'Max': f"{specs.max():.4f}"
            })
        
        cellex_df = pd.DataFrame(cellex_stats)
        
        ax1.axis('off')
        table1 = ax1.table(cellText=cellex_df.values, colLabels=cellex_df.columns,
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table1.auto_set_font_size(False)
        table1.set_fontsize(9)
        table1.scale(1, 2)
        
        # Color header
        for i in range(len(cellex_df.columns)):
            table1[(0, i)].set_facecolor('#4CAF50')
            table1[(0, i)].set_text_props(weight='bold', color='white')
        
        ax1.set_title('CELLEX: Cell-Type Specificity Statistics',
                     fontsize=13, fontweight='bold', pad=20)
        
        # ===== RIGHT PANEL: CELLECT Statistics =====
        cellect_stats = []
        for cell_type in self.enrichment_df.index:
            row = self.enrichment_df.loc[cell_type]
            sig = "***" if row['enrichment_p'] < 0.05 else "ns"
            cellect_stats.append({
                'Cell Type': cell_type,
                'P-value': f"{row['enrichment_p']:.2e}",
                '-log10(P)': f"{row['log10_p']:.4f}",
                'Sig': sig
            })
        
        cellect_df = pd.DataFrame(cellect_stats)
        
        ax2.axis('off')
        table2 = ax2.table(cellText=cellect_df.values, colLabels=cellect_df.columns,
                          cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1, 2)
        
        # Color header
        for i in range(len(cellect_df.columns)):
            table2[(0, i)].set_facecolor('#2196F3')
            table2[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color significant rows
        for i in range(1, len(cellect_df) + 1):
            if cellect_df.iloc[i-1]['Sig'] == '***':
                for j in range(len(cellect_df.columns)):
                    table2[(i, j)].set_facecolor('#ffcccc')
        
        ax2.set_title('CELLECT: Genetic Enrichment Statistics',
                     fontsize=13, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '8_Summary_Statistics.png', 
                   dpi=300, bbox_inches='tight')
        logger.info(f"  ✓ Saved: 8_Summary_Statistics.png")
        plt.close()
    
    # ========================================================================
    # RUN ALL VISUALIZATIONS
    # ========================================================================
    
    def create_all_visualizations(self):
        """Create all 8 visualizations"""
        logger.info("\n" + "🎨" * 50)
        logger.info("CREATING ALL VISUALIZATIONS FROM CELLEX & CELLECT")
        logger.info("🎨" * 50)
        
        self.load_data()
        self.plot_cellect_enrichment()
        self.plot_cellex_specificity_heatmap_top_genes(top_n=40)
        self.plot_cellex_specificity_by_celltype()
        self.plot_cellex_specificity_boxplot()
        self.plot_celltype_specificity_patterns()
        self.plot_enrichment_vs_specificity()
        self.plot_top_genes_per_celltype(top_n=5)
        self.create_summary_statistics_figure()
        
        logger.info("\n" + "=" * 100)
        logger.info("✓ ALL VISUALIZATIONS CREATED!")
        logger.info("=" * 100)
        logger.info(f"\nSaved to: {self.output_dir}")
        logger.info(f"\nGenerated files:")
        logger.info(f"  1. 1_CELLECT_Enrichment.png")
        logger.info(f"  2. 2_CELLEX_Specificity_Heatmap_Top40.png")
        logger.info(f"  3. 3_CELLEX_Specificity_Distribution.png")
        logger.info(f"  4. 4_CELLEX_Specificity_Boxplot.png")
        logger.info(f"  5. 5_CellType_Similarity_Patterns.png")
        logger.info(f"  6. 6_Enrichment_vs_Specificity.png")
        logger.info(f"  7. 7_Top5_Genes_Per_CellType.png")
        logger.info(f"  8. 8_Summary_Statistics.png")

if __name__ == "__main__":
    viz = CELLECTCELLEXVisualizations(
        specificity_file="cellex_output/T2D_celltype_specificity_index.tsv",
        enrichment_file="cellect_output/T2D_celltype_enrichment_results.tsv",
        output_dir="visualizations/"
    )
    viz.create_all_visualizations()