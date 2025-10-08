from nerthus import ImageProcessor, ensure_directory, setup_logging

def get_data_path() -> str:
    import kagglehub
    return kagglehub.dataset_download("waltervanhuissteden/the-nerthus-dataset")

def main() -> None:
    logger = setup_logging()

    data_path = get_data_path()
    logger.info(f"Dataset downloaded to: {data_path}")

    processor = ImageProcessor(data_path=data_path)

    logger.info("Discovering image files")
    image_files = processor.discover_image_files()

    logger.info("Extract an image metadata")
    metadata = processor.extract_image_metadata(image_path=image_files['0'][0])
    for k, v in metadata.items():
        print(f"-- {k}: {v}")
    
    logger.info("Analyzing an image features")
    image = processor.load_image(image_path=image_files['0'][0])
    features = processor.analyze_image_features(image=image)
    for k, v in features.items():
        print(f"-- {k}: {v}")
    
    
    logger.info("Creating image montage for each class")
    for class_label, images in image_files.items():
        images = [processor.load_image(image_path=image_path) for image_path in images]
        processor.create_image_montage(
            images=images,
            output_path=f"images/sample_images_montage_{class_label}.png",
            max_images=12
        )


    logger.info("Saving sample images for each class")
    processor.save_sample_images(
        image_files=image_files,
        samples_per_class=5,
        output_dir='images/sample_images'
    )

if __name__ == "__main__":
    ensure_directory("images")    
    main()

    print("\nSee images/ for results.\n")