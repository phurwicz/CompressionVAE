from cvae import cvae
import numpy as np


def main():
    X = np.random.uniform(size=(5000,300))
    embedder = cvae.CompressionVAE(X)
    embedder.train()
    Z = embedder.embed(X)
    return Z

if __name__ == '__main__':
    Z = main()
    assert Z.shape == (5000,2)
