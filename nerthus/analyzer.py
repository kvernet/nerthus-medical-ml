import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend FIRST
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import kagglehub
from scipy import stats
import warnings
from tqdm import tqdm

from .utils import setup_logging, ensure_directory, save_dataframe
from .processor import ImageProcessor

warnings.filterwarnings('ignore')

class NerthusAnalyzer:
    """
    A comprehensive analyzer for the Nerthus medical image dataset.
    
    The Nerthus dataset contains colonoscopy images with different bowel preparation
    quality scores (BBPS 0-3) for medical image analysis research.
    """
    
    def __init__(self, data_path: Optional[str] = None, 
                 random_state: int = 69,
                 output_dir: str = "outputs/analysis"):
        """
        Initialize the Nerthus analyzer for image data.
        
        Args:
            data_path: Path to the dataset directory. If None, downloads from Kaggle.
            output_dir: Directory to save analysis outputs.
            images_dir: Directory to save generated images.
        """
        np.random.seed(random_state)
        self.logger = setup_logging()
        self.output_dir = output_dir
        self.random_state = random_state
        self.images_dir = f"{output_dir}/images"
        self.data_path = data_path
        self.image_processor = None
        self.image_files = {}
        self.dataset_metadata = {}
        self.image_features = pd.DataFrame()
        
        # Create directories
        ensure_directory(output_dir)
        ensure_directory(self.images_dir)
        ensure_directory(os.path.join(self.images_dir, 'sample_images'))
        
        self.logger.info("NerthusAnalyzer initialized for image analysis")
    
    def load_data(self) -> Dict[str, List[str]]:
        """
        Load the Nerthus image dataset from the specified path or download from Kaggle.
        
        Returns:
            Dictionary of image files organized by BBPS class
        """
        self.logger.info("Loading Nerthus image dataset...")
        
        if self.data_path is None:
            self.logger.info("Downloading dataset from Kaggle...")
            self.data_path = kagglehub.dataset_download("waltervanhuissteden/the-nerthus-dataset")
            self.logger.info(f"Dataset downloaded to: {self.data_path}")
        
        # Initialize image processor and discover files
        self.image_processor = ImageProcessor(self.data_path)
        self.image_files = self.image_processor.discover_image_files()
        
        if not self.image_files:
            # Try specific path structure
            specific_path = os.path.join(self.data_path, "nerthus-dataset-frames", "nerthus-dataset-frames")
            if os.path.exists(specific_path):
                self.logger.info(f"Trying specific path: {specific_path}")
                self.image_processor = ImageProcessor(specific_path)
                self.image_files = self.image_processor.discover_image_files()
        
        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {self.data_path}")
        
        total_images = sum(len(images) for images in self.image_files.values())
        self.logger.info(f"Found {total_images} images across {len(self.image_files)} classes")
        
        # Store basic metadata
        self.dataset_metadata['total_images'] = total_images
        self.dataset_metadata['classes'] = list(self.image_files.keys())
        self.dataset_metadata['images_per_class'] = {cls: len(images) for cls, images in self.image_files.items()}
        
        return self.image_files
    
    def extract_image_metadata(self) -> pd.DataFrame:
        """
        Extract metadata from all image files.
        
        Returns:
            DataFrame containing image metadata
        """
        if not self.image_files:
            self.load_data()
        
        self.logger.info("Extracting image metadata...")
        
        all_metadata = []
        
        for class_label, image_paths in self.image_files.items():
            for image_path in tqdm(image_paths, desc=f"Processing class {class_label}"):
                metadata = self.image_processor.extract_image_metadata(image_path)
                metadata['bbps_class'] = class_label
                all_metadata.append(metadata)
        
        metadata_df = pd.DataFrame(all_metadata)
        
        # Save metadata
        save_dataframe(metadata_df, os.path.join(self.output_dir, 'image_metadata.csv'))
        
        self.logger.info(f"Extracted metadata for {len(metadata_df)} images")
        return metadata_df
    
    def analyze_image_features(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Analyze features from images.
        
        Args:
            sample_size: Number of images to sample per class (None for all)
            
        Returns:
            DataFrame containing image features
        """
        if not self.image_files:
            self.load_data()
        
        self.logger.info("Analyzing image features...")
        
        all_features = []
        
        for class_label, image_paths in self.image_files.items():
            size = len(image_paths)
            # Sample images if specified
            if sample_size and sample_size < len(image_paths):
                size = sample_size
            # Get a random image_paths of size=size
            sample_paths = np.random.choice(image_paths, size, replace=False)
            
            for image_path in tqdm(sample_paths, desc=f"Analyzing class {class_label}"):
                # Load and analyze image
                image = self.image_processor.load_image(image_path)
                if image is not None:
                    features = self.image_processor.analyze_image_features(image)
                    if features:
                        features.update({
                            'file_path': image_path,
                            'bbps_class': class_label,
                            'file_name': os.path.basename(image_path)
                        })
                        all_features.append(features)
        
        self.image_features = pd.DataFrame(all_features)
        
        # Save image features
        save_dataframe(self.image_features, os.path.join(self.output_dir, 'image_features.csv'))
        
        self.logger.info(f"Analyzed {len(self.image_features)} images")
        return self.image_features
    
    def create_dataset_overview(self, metadata_df: pd.DataFrame) -> None:
        """
        Create visualizations showing dataset overview.
        
        Args:
            metadata_df: DataFrame containing image metadata
        """
        self.logger.info("Creating dataset overview visualizations...")
        
        # 1. Class distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        class_counts = metadata_df['bbps_class'].value_counts().sort_index()
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        
        bars = plt.bar(class_counts.index, class_counts.values, color=colors)
        plt.title('Image Distribution by BBPS Class')
        plt.xlabel('BBPS Class')
        plt.ylabel('Number of Images')
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', va='bottom')
        
        # 2. Image size distribution
        plt.subplot(1, 2, 2)
        if 'width' in metadata_df.columns and 'height' in metadata_df.columns:
            plt.scatter(metadata_df['width'], metadata_df['height'], alpha=0.6)
            plt.title('Image Dimensions Distribution')
            plt.xlabel('Width (pixels)')
            plt.ylabel('Height (pixels)')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'dataset_overview.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. File size distribution
        plt.figure(figsize=(10, 6))
        if 'file_size' in metadata_df.columns:
            # Convert to MB
            file_sizes_mb = metadata_df['file_size'] / (1024 * 1024)
            plt.hist(file_sizes_mb.dropna(), bins=30, alpha=0.7, edgecolor='black')
            plt.title('Distribution of Image File Sizes')
            plt.xlabel('File Size (MB)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.images_dir, 'file_size_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_feature_analysis_plots(self) -> None:
        """
        Create visualizations analyzing image features.
        """
        if self.image_features.empty:
            self.logger.warning("No image features available. Run analyze_image_features() first.")
            return
        
        self.logger.info("Creating feature analysis visualizations...")
        
        numeric_features = self.image_features.select_dtypes(include=[np.number]).columns
        # Remove non-feature columns
        non_feature_cols = ['bbps_class']
        numeric_features = [f for f in numeric_features if f not in non_feature_cols]
        
        # Analyze first few numeric features
        plot_features = numeric_features[:9]  # Limit to first 6 features
        
        n_rows = 3
        n_cols = 3
        _, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(plot_features):
            if i < len(axes):
                # Boxplot by BBPS class
                data_to_plot = []
                classes = sorted(self.image_features['bbps_class'].unique())
                
                for cls in classes:
                    class_data = self.image_features[self.image_features['bbps_class'] == cls][feature].dropna()
                    data_to_plot.append(class_data)
                
                box_plot = axes[i].boxplot(data_to_plot, labels=classes, patch_artist=True)
                
                # Add colors to boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
                for patch, color in zip(box_plot['boxes'], colors):
                    patch.set_facecolor(color)
                
                axes[i].set_title(f'{feature} by BBPS Class', fontsize=12)
                axes[i].set_xlabel('BBPS Class', fontsize=10)
                axes[i].set_ylabel(feature, fontsize=10)
                axes[i].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(len(plot_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.images_dir, 'feature_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation heatmap of features
        if len(plot_features) > 1:
            correlation_matrix = self.image_features[plot_features].corr()
            
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
            plt.title('Correlation Matrix of Image Features', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.images_dir, 'feature_correlations.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_sample_montage(self, images_per_class: int = 4) -> None:
        """
        Create a montage of sample images from each class.
        
        Args:
            images_per_class: Number of images to sample from each class
        """
        if not self.image_files:
            self.load_data()
        
        self.logger.info("Creating sample image montage...")
        
        all_sample_images = []
        image_titles = []
        
        for class_label, image_paths in self.image_files.items():
            # Sample images from this class
            sample_paths = image_paths[:images_per_class]
            
            for image_path in sample_paths:
                # Load image
                image = self.image_processor.load_image(image_path)
                if image is not None:
                    all_sample_images.append(image)
                    image_name = os.path.splitext(os.path.basename(image_path))[0]
                    # Shorten long names
                    if len(image_name) > 5:
                        image_name = image_name[:5] + "..."
                    image_titles.append(f"Class {class_label}\n{image_name}")
        
        # Create montage
        if all_sample_images:
            montage_path = os.path.join(self.images_dir, 'sample_images_montage.png')
            self.image_processor.create_image_montage(all_sample_images, montage_path, image_titles)
            self.logger.info(f"Sample montage saved to: {montage_path}")
            
            # Also save individual sample images
            sample_output_dir = os.path.join(self.images_dir, 'sample_images')
            self.image_processor.save_sample_images(self.image_files, samples_per_class=images_per_class, output_dir=sample_output_dir)
    
    def perform_statistical_analysis(self) -> Dict[str, Any]:
        """
        Perform statistical analysis on image features.
        
        Returns:
            Dictionary containing statistical analysis results
        """
        if self.image_features.empty:
            self.logger.warning("No image features available. Run analyze_image_features() first.")
            return
        
        self.logger.info("Performing statistical analysis...")
        
        statistical_results = {}
        numeric_features = self.image_features.select_dtypes(include=[np.number]).columns
        # Remove non-feature columns
        non_feature_cols = []
        numeric_features = [f for f in numeric_features if f not in non_feature_cols]
        
        if len(numeric_features) > 0:
            # Descriptive statistics
            statistical_results['descriptive_stats'] = self.image_features[numeric_features].describe().to_dict()
            
            # ANOVA test to see if features differ by BBPS class
            statistical_results['anova_results'] = {}
            
            for feature in numeric_features:
                groups = []
                for cls in sorted(self.image_features['bbps_class'].unique()):
                    group_data = self.image_features[self.image_features['bbps_class'] == cls][feature].dropna()
                    if len(group_data) > 1:  # Need at least 2 samples per group
                        groups.append(group_data)
                
                if len(groups) >= 2:
                    try:
                        f_stat, p_value = stats.f_oneway(*groups)
                        statistical_results['anova_results'][feature] = {
                            'f_statistic': f_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except Exception as e:
                        statistical_results['anova_results'][feature] = {
                            'f_statistic': None,
                            'p_value': None,
                            'significant': False,
                            'error': str(e)
                        }
        
        # Save statistical results
        if statistical_results.get('descriptive_stats'):
            stats_df = pd.DataFrame(statistical_results['descriptive_stats'])
            save_dataframe(stats_df, os.path.join(self.output_dir, 'statistical_summary.csv'))
        
        self.logger.info("Statistical analysis completed")

        # Save ANOVA results
        if statistical_results.get('anova_results'):
            anova_df = pd.DataFrame(statistical_results['anova_results'])
            save_dataframe(anova_df, os.path.join(self.output_dir, 'anova_results.csv'))
        
        self.logger.info("ANOVA test completed")
        return statistical_results
    
    def generate_report(self, sample_size: int=100, images_per_class: int=4) -> str:
        """
        Generate a comprehensive analysis report for the image dataset.
        
        Returns:
            String containing the analysis report
        """
        if not self.image_files:
            self.load_data()
        
        self.logger.info("Generating comprehensive analysis report...")
        
        # Perform all analyses
        metadata_df = self.extract_image_metadata()
        features_df = self.analyze_image_features(sample_size=sample_size)
        statistical_results = self.perform_statistical_analysis()
        
        # Create visualizations
        self.create_dataset_overview(metadata_df=metadata_df)
        self.create_feature_analysis_plots()
        self.create_sample_montage(images_per_class=images_per_class)
        
        # Generate report
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("NERTHUS MEDICAL IMAGE DATASET ANALYSIS REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"\nGenerated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Dataset Overview
        report_lines.append("\n1. DATASET OVERVIEW")
        report_lines.append("-" * 40)
        report_lines.append(f"Total images: {self.dataset_metadata['total_images']}")
        report_lines.append(f"BBPS classes: {', '.join(self.dataset_metadata['classes'])}")
        report_lines.append("Images per class:")
        for cls, count in self.dataset_metadata['images_per_class'].items():
            report_lines.append(f"  - Class {cls}: {count} images")
        
        # Image Metadata Summary
        report_lines.append("\n2. IMAGE METADATA SUMMARY")
        report_lines.append("-" * 40)
        if not metadata_df.empty:
            report_lines.append(f"Average image size: {metadata_df['width'].mean():.0f}x{metadata_df['height'].mean():.0f} pixels")
            report_lines.append(f"Common file format: {metadata_df['file_extension'].mode().iloc[0]}")
            if 'file_size' in metadata_df.columns:
                avg_size_mb = metadata_df['file_size'].mean() / (1024 * 1024)
                report_lines.append(f"Average file size: {avg_size_mb:.2f} MB")
        
        # Feature Analysis
        report_lines.append("\n3. FEATURE ANALYSIS")
        report_lines.append("-" * 40)
        if not features_df.empty:
            numeric_features = features_df.select_dtypes(include=[np.number]).columns
            numeric_features = [f for f in numeric_features if f not in ['bbps_class']]
            report_lines.append(f"Number of features extracted: {len(numeric_features)}")
            report_lines.append(f"Key medical image features: {', '.join(numeric_features[:8])}")
        
        # Statistical Significance
        report_lines.append("\n4. STATISTICAL SIGNIFICANCE")
        report_lines.append("-" * 40)
        if statistical_results.get('anova_results'):
            significant_features = [
                feature for feature, results in statistical_results['anova_results'].items()
                if results.get('significant', False)
            ]
            report_lines.append(f"Features significantly different by BBPS class: {len(significant_features)}")
            if significant_features:
                report_lines.append("Significant features:")
                for feature in significant_features[:5]:  # Show top 5
                    p_val = statistical_results['anova_results'][feature]['p_value']
                    report_lines.append(f"  - {feature} (p={p_val:.4f})")
        
        # Medical Research Implications
        report_lines.append("\n5. MEDICAL RESEARCH IMPLICATIONS")
        report_lines.append("-" * 40)
        report_lines.append("• Dataset suitable for bowel preparation quality assessment")
        report_lines.append("• BBPS scores enable supervised learning for quality classification")
        report_lines.append("• Features like edge density and texture can inform quality assessment")
        report_lines.append("• Potential for computer-aided diagnosis systems")
        report_lines.append("• Can be used for training medical AI models")
        
        # Technical Recommendations
        report_lines.append("\n6. TECHNICAL RECOMMENDATIONS")
        report_lines.append("-" * 40)
        report_lines.append("• Use CNN architectures for image classification")
        report_lines.append("• Consider transfer learning with medical image models")
        report_lines.append("• Implement data augmentation for medical images")
        report_lines.append("• Use appropriate evaluation metrics for medical AI")
        report_lines.append("• Consider class imbalance in BBPS scores")
        
        report_lines.append("\n" + "=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)
        
        report_content = "\n".join(report_lines)
        
        # Save report to file
        report_path = os.path.join(self.output_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.logger.info(f"Analysis report saved to: {report_path}")
        return report_content
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a quick summary of the dataset and analysis.
        
        Returns:
            Dictionary containing summary information
        """
        if not self.image_files:
            self.load_data()
        
        summary = {
            'dataset_name': 'Nerthus Medical Image Dataset',
            'total_images': self.dataset_metadata['total_images'],
            'bbps_classes': self.dataset_metadata['classes'],
            'images_per_class': self.dataset_metadata['images_per_class'],
            'description': 'Colonoscopy images with bowel preparation quality scores (BBPS 0-3)'
        }
        
        return summary

def main():
    """Main entry point for the nerthus-analyze command."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Nerthus Medical Analysis')

    parser.add_argument(
        "-s", "--random_state",
        type=int,
        default=42,
        help="Random state (default: 42)"
    )

    parser.add_argument(
        "-p", "--samples",
        type=int,
        default=200,
        help="Sample size per class (default: 200)"
    )

    parser.add_argument(
        "-n",
        type=int,
        default=4,
        help="Number of images per class (default: 4)"
    )

    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        default="outputs/analysis/",
        help="Output directory (default: outputs/analysis/)"
    )
    
    args = parser.parse_args()
    
    from .analyzer import NerthusAnalyzer
    analyzer = NerthusAnalyzer(
        random_state=args.random_state,
        output_dir=args.output_dir)
    
    print("Nerthus Medical Analysis")
    print("=" * 40)
    
    # Load data
    analyzer.load_data()

    # Get summary
    summary = analyzer.get_summary()
    print(f"Dataset Summary:")
    print(f"  Dataset name    : {summary['dataset_name']}")
    print(f"  Total images    : {summary['total_images']}")
    print(f"  BBPS classes    : {summary['bbps_classes']}")
    print(f"  Images per class: {summary['images_per_class']}")
    print(f"  Description     : {summary['description']}")
    
    # Generate comprehensive report
    print("\nGenerating analysis report...")
    analyzer.generate_report(sample_size=args.samples, images_per_class=args.n)
    
    print(f"\nAnalysis completed!")
    print(f"Check {args.output_dir} directory for results.")

if __name__ == "__main__":
    main()