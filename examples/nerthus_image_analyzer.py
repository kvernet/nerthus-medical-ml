from nerthus import NerthusAnalyzer

def main():
    analyzer = NerthusAnalyzer()
    
    print("Nerthus Medical ML - Analysis Mode")
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
    analyzer.generate_report()
    
    print(f"\nAnalysis completed!")
    print(f"Check outputs/ and outputs/images directories for results.")

if __name__ == "__main__":
    main()