
from utils.preprocessing import load_and_preprocess_images
from utils.feature_extraction import extract_features
from utils.pso_selection import select_features_pso
from utils.classification import train_and_evaluate
from utils.visualization import visualize_tsne

print("🔧 Loading images...")
X, y = load_and_preprocess_images()

print("📡 Extracting features...")
features = extract_features(X)

print("🐝 Running PSO...")
mask = select_features_pso(features, y)

print("🎯 Training classifier...")
train_and_evaluate(features[:, mask], y)

print("🔍 Visualizing t-SNE...")
visualize_tsne(features[:, mask], y)