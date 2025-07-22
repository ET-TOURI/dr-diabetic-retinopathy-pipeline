from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_tsne(features, labels):
    tsne = TSNE(n_components=2, random_state=42)
    emb = tsne.fit_transform(features)
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap='plasma')
    plt.title("t-SNE of Selected Features")
    plt.show()