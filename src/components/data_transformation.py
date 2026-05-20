import pandas as pd
import numpy as np
import re
import os
import sys

from dataclasses import dataclass
from nltk.corpus import stopwords
from nltk import PorterStemmer, sent_tokenize
from gensim.models import Word2Vec, KeyedVectors
from tqdm import tqdm
from gensim.utils import simple_preprocess

from sklearn.preprocessing import LabelEncoder


from src.utils.exception import CustomException
from src.utils.logger import get_logger
from src.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    Vectorizer_path: str = os.path.join('artifacts', 'vectorizer.pkl')
    encoder_path: str = os.path.join('artifacts', 'encoder.pkl')


logger = get_logger(__name__)

class TextCleaner:
    """ Stateless text cleaning"""

    def __init__(self):
        self.sw_list = stopwords.words('english')
        self.ps = PorterStemmer()

    def remove_html_tags(self, text):
        return re.sub(re.compile('<.*?>'), '', text)
    
    def to_lowercase(self, text):
        return text.lower()

    def remove_stopwords(self, text):
        tokens = [w for w in text.split() if w not in self.sw_list]
        return " ".join(tokens)
    
    def stem(self, text):
        return " ".join([self.ps.stem(w) for w in text.split()])
    
    def clean(self, text):
        text = self.remove_html_tags(text)
        text = self.to_lowercase(text)
        text = self.remove_stopwords(text)
        text = self.stem(text)

        return text
    

class Vectorizer:
    def __init__(self, window=5, min_count=2, vector_size=100):
        self.window = window
        self.min_count = min_count
        self.vector_size = vector_size
        # self.model = None

    
    def _build_sentences(self, texts):
        story = []
        for doc in texts:
            for sent in sent_tokenize(doc):
                story.append(simple_preprocess(sent))
            logger.info("Word Tokenization completed!")
        return story
    
    def fit(self, X_train_texts):
        sentences = self._build_sentences(X_train_texts)
        self.model = Word2Vec(
            window=self.window,
            min_count=self.min_count,
            vector_size=self.vector_size
        )

        self.model.build_vocab(sentences)
        self.model.train(
            sentences,
            total_examples=self.model.corpus_count,
            epochs = self.model.epochs
        )
        return self
    
    def _document_vector(self, doc):
        tokens = [w for w in doc.split() if w in self.model.wv.key_to_index]
        logger.info(f"{tokens}")
        if not tokens:
            return np.zeros(self.model.vector_size)
        return  np.mean(self.model.wv[tokens], axis=0)
    
    def transform(self, texts):
        return np.array([self._document_vector(doc) for doc in texts])
    
    def fit_transform(self, X_train_texts):
        self.fit(X_train_texts)
        return self.transform(X_train_texts)
    




class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiateDataTransformation(self, train_path, test_path):
        logger.info("Data transformation started.")

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info(f"Train: {train_df.shape} | Test: {test_df.shape}")

            cleaner = TextCleaner()
            train_df['review'] = train_df['review'].apply(cleaner.clean)
            test_df['review'] = test_df['review'].apply(cleaner.clean)
            logger.info("Text cleaning complete.")

            
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(train_df['sentiment'])
            y_test = encoder.transform(test_df['sentiment'])
            logger.info("Label encoded successfully.")


            vectorizer = Vectorizer()
            X_train = vectorizer.fit_transform(train_df['review'].values)
            logger.info("Vectorization completed in training data.")
            X_test = vectorizer.transform(test_df['review'].values)
            logger.info(f"Vectorization complete. X_train: {X_train.shape} | X_test: {X_test.shape}")

            os.makedirs('artifacts', exist_ok=True)
            save_object(self.config.Vectorizer_path, vectorizer)
            save_object(self.config.encoder_path, encoder)
            logger.info(f"Vectorizer saved to: {self.config.Vectorizer_path}")

            logger.info("Data Transformation completed.")
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            raise CustomException(e, sys)

