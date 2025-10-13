
def predict(file_path, img_class):
    from loader import modelLoader
    from PIL import Image

    # Load models
    modelLoader.load_models("XGBoost.joblib", "CNN.keras")

    # Load image
    img = Image.open(file_path)
    ml_pred, ml_conf, ml_msg = modelLoader.predict_with_ml(img)
    cnn_pred, cnn_conf, cnn_msg = modelLoader.predict_with_cnn(img, debug=True)

    print("Class     :", img_class)
    
    print("ML prediction")
    print("Prediction:", ml_pred)
    print("Confidence:", ml_conf)
    print("Message   :", ml_msg)
    
    print("CNN prediction")
    print("Prediction:", cnn_pred)
    print("Confidence:", cnn_conf)
    print("Message   :", cnn_msg)

def main():
    for i, image_path in enumerate([
        "static/images/0/bowel_2_score_0-0_00000052.jpg",
        "static/images/1/bowel_7_score_1-1_00000064.jpg",
        "static/images/2/bowel_14_score_2-0_00000003.jpg",
        "static/images/3/bowel_17_score_3-0_00000023.jpg"
    ]):
        predict(image_path, i)

if __name__ == "__main__":
    main()