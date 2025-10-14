"""
Command line interface for Nerthus Medical ML
"""

def main():
    """Main entry point for the nerthus command."""
    import argparse
    parser = argparse.ArgumentParser(description="Run Nerthus with optional ML features.")

    parser.add_argument(
        '--analysis', 
        action='store_true',
        help='Use the analysis component (only with --analysis).'
    )
    
    parser.add_argument(
        '--cnn', 
        action='store_true',
        help='Use the CNN model (only with --cnn).'
    )
    
    parser.add_argument(
        '--ml', 
        action='store_true',
        help='Use machine learning mode (only with --ml).'
    )

    parser.add_argument(
        '--processor', 
        action='store_true',
        help='Use the processor component (only with --processor).'
    )

    args = parser.parse_args()
    
    import subprocess

    # --- Use the parsed arguments ---
    if args.analysis:
        subprocess.run(["nerthus-analyze"])
    elif args.cnn:
        subprocess.run(["nerthus-cnn"])
    elif args.ml:
        subprocess.run(["nerthus-ml"])
    elif args.processor:
        subprocess.run(["nerthus-processor"])
    else:
        print("Error, run with --help for help")

if __name__ == "__main__":
    main()