"""
Nerthus Analyzer:
    Loading images
    Getting summary
    Generating report
"""

from nerthus import NerthusAnalyzer

def main():
    output_dir="outputs/analysis/"
    analyzer = NerthusAnalyzer(
        random_state=42,
        output_dir=output_dir)
    
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
    analyzer.generate_report(
        sample_size=200,
        images_per_class=4
    )
    
    print(f"\nAnalysis completed!")
    print(f"Check {output_dir} directory for results.")

if __name__ == "__main__":
    main()