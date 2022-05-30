import logging
import numpy as np
import pickle
import io
from PIL import Image
from typing import List, Tuple
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

MODEL_PATH = "solver/model/solverModel.pkl"

logging.basicConfig(level=logging.INFO)

class SolverModel:
    __fitted: bool = False
    __created: bool = False
    __clf: KNeighborsClassifier = None

    def __init__(self) -> None:
        pass

    def loadModel(self) -> None:
        with open(MODEL_PATH, "rb") as f:
            self.__clf = pickle.load(f)
        self.__created = True
        self.__fitted = True

    def saveModel(self) -> None:
        if self.__created and self.__fitted:
            with open(MODEL_PATH, "wb") as f:
                pickle.dump(self.__clf, f)
            logging.info(f"Model Saved: {MODEL_PATH}")
        else:
            raise Exception("Model not created or not fitted")

    def createModel(self, k: int = 5, knnP: int = 2) -> None:
        self.__created = True
        self.__clf = KNeighborsClassifier(n_neighbors=k, p=knnP)
        logging.info("Model Created")

    def predict(self, data: List[float]) -> str:
        if not self.__fitted:
            raise Exception("Model not fitted")
        return self.__clf.predict(data)
    
    def fitModel(self) -> None:
        if not self.__created: 
            raise Exception("Model not created")
        if self.__fitted:
            raise Exception("Model already fitted")
        X_train, y_train, X_test, y_test = SolverModel.prepareTrainSet()
        self.__clf.fit(X_train, y_train)
        logging.info("Model fitted")
        self.__fitted = True
        logging.info(f"Model Score: {self.__clf.score(X_test, y_test)}")

    @staticmethod
    def prepareTrainSet() -> Tuple[List[List[float]], list[str]]:
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train/255
        X_train_reshaped = X_train.reshape(len(X_train), 28*28)
        X_test = X_test/255
        X_test_reshaped = X_test.reshape(len(X_test), 28*28)
        return X_train_reshaped, y_train, X_test_reshaped, y_test
    
    def predictUploadedImage(self, imagePath) -> int:
        with open(imagePath, "rb") as f:
            readBytes = f.read()
        img = Image.open(io.BytesIO(readBytes)).convert('L')
        input_size = 280
        output_size = 28
        bin_size = input_size // output_size
        small_image = np.asarray(img).reshape((output_size, bin_size, output_size, bin_size, 1)).min(3).min(1)
        topredict = ((np.asarray(small_image)*-1+255)/255).reshape(1, 28*28)
        image = small_image
        return self.__clf.predict(topredict)[0]

    @staticmethod
    def plotImage(image: np.ndarray) -> None:
        fig = plt.figure()
        plt.imshow(image,cmap='gray')
        plt.show()
        
# def main():
#     sm = SolverModel()
#     sm.createModel()
#     sm.fitModel()
#     sm.saveModel()


# if __name__ == "__main__":
#     main()