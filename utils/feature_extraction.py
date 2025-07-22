from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import preprocess_input

def extract_features(X):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    X_processed = preprocess_input(X)
    return model.predict(X_processed, verbose=1)