import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MessageToVector:
    def __init__(self, n_components=100):
        """
        Initializes the MessageToVector model with the specified number of principal components.

        Parameters:
        n_components (int): The number of principal components to retain after applying PCA. Default is 100.

        Attributes:
        vocabulary (dict): A dictionary mapping words to their corresponding indices.
        inverse_vocabulary (list): A list of words corresponding to their indices.
        n_components (int): The number of principal components to retain after applying PCA.
        pca (PCA object): The PCA model to reduce dimensionality of the vectors. Initialized to None.
        """
        self.vocabulary = {}
        self.inverse_vocabulary = []
        self.n_components = n_components
        self.pca = None
        self.scaler = None

    def fit(self, corpus):
        """
        Fits the model with the corpus by creating the vocabulary.

        Parameters:
        corpus (list of str): A list of strings to build the vocabulary.
        """
        unique_words = set()
        for document in corpus:
            words = document.lower().split()
            unique_words.update(words)

        self.vocabulary = {word: idx for idx, word in enumerate(sorted(unique_words))}
        self.inverse_vocabulary = sorted(unique_words)

    def transform_to_binary(self, text):
        """
        Transforms a given text into a binary vector based on the fitted vocabulary.

        Parameters:
        text (str): A string to be transformed.

        Returns:
        np.ndarray: The binary vectorized representation of the string.
        """
        vector = np.zeros(len(self.vocabulary), dtype=int)
        words = text.lower().split()
        for word in words:
            if word in self.vocabulary:
                vector[self.vocabulary[word]] += 1
        return vector

    def fit_transform_to_reduced(self, corpus):
        """
        Fits the model with the corpus and transforms the corpus into binary vectors.
        Then applies PCA to reduce the dimensionality.

        Parameters:
        corpus (list of str): A list of strings to build the vocabulary.
        n_components (int): The number of principal components to keep.

        Returns:
        np.ndarray: The PCA-reduced binary vectorized representation of the corpus.
        """
        self.fit(corpus)
        binary_vectors = np.array([self.transform_to_binary(document) for document in corpus])
        self.scaler = StandardScaler()
        scaled_vectors = self.scaler.fit_transform(binary_vectors)
        # Apply PCA to reduce dimensionality
        self.pca = PCA(n_components=self.n_components)
        reduced_vectors = self.pca.fit_transform(scaled_vectors)
        return reduced_vectors

    def transform_to_reduce_new(self, text):
        """
        Transforms a new text into a binary vector and then applies PCA.

        Parameters:
        text (str): A string to be transformed.

        Returns:
        np.ndarray: The PCA-reduced binary vectorized representation of the string.
        """
        binary_vector = self.transform(text)
        if self.scaler is not None and self.pca is not None:
            scaled_vectors = self.scaler.transform([binary_vector])
            reduced_vector = self.pca.transform(scaled_vectors)
            return reduced_vector[0]
        else:
            raise ValueError("PCA have not been fitted. Call fit_transform_to_reduced first.")

    def get_feature_names(self):
        """
        Returns the feature names (vocabulary) learned by the vectorizer.

        Returns:
        list of str: The vocabulary of the corpus.
        """
        return self.inverse_vocabulary

