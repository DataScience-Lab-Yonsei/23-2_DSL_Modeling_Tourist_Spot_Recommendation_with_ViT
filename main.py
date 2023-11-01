import argparse

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from model import *
from index import *

MODEL_DIR = "./pretrained"
DATABASE = np.load("./database/database.npz")["arr_0"]
ORIGIN = np.load("./database/origin.npz")["arr_0"]

def config():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", default="local", type=str, help="mode")
    p.add_argument("--search_mode", default="faiss", type=str, help="search mode")
    p.add_argument("--image_dir", default="./image", help="directory of image")
    p.add_argument("--image_url", default=None, help="url of image")

    return p.parse_args()

def recommend(config):
    model, feature_extractor = pretrained(MODEL_DIR=MODEL_DIR)

    if config.mode == "local":
        abroad = Image.open(config.image_dir)
        inputs = feature_extractor(images=abroad, return_tensors="pt")
        represent = model(**inputs).hidden_states[-1][0][-1].detach().numpy()

        if config.search_mode == "faiss":
            index = index_faiss(DATABASE)
            _, indices = index.search(represent.reshape(1, -1), 1)
            
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(abroad)
            ax1.set_title("Abroad")
            ax1.axis("off")

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(ORIGIN[indices[0][0]])
            ax2.set_title("Recommendation")
            ax2.axis("off")

            plt.show()

        elif config.search_mode == "hnsw":
            p = index_hnsw(DATABASE)
            indices, _ = p.knn_query(represent, k=1)

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(abroad)
            ax1.set_title("Abroad")
            ax1.axis("off")

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(ORIGIN[indices[0][0]])
            ax2.set_title("Recommendation")
            ax2.axis("off")

            plt.show()

    elif config.mode == "remote":
        import requests
        from io import BytesIO

        response = requests.get(config.image_url)
        abroad = Image.open(BytesIO(response.content))
        inputs = feature_extractor(images=abroad, return_tensors="pt")
        represent = model(**inputs).hidden_states[-1][0][-1].detach().numpy()

        if config.search_mode == "faiss":
            index = index_faiss(DATABASE)
            _, indices = index.search(represent.reshape(1, -1), 1)
            
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(abroad)
            ax1.set_title("Abroad")
            ax1.axis("off")

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(ORIGIN[indices[0][0]])
            ax2.set_title("Recommendation")
            ax2.axis("off")

            plt.show()

        elif config.search_mode == "hnsw":
            p = index_hnsw(DATABASE)
            indices, _ = p.knn_query(represent, k=1)
            
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.imshow(abroad)
            ax1.set_title("Abroad")
            ax1.axis("off")

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.imshow(ORIGIN[indices[0][0]])
            ax2.set_title("Recommendation")
            ax2.axis("off")

            plt.show()

if __name__ == "__main__":
    config = config()
    recommend(config)