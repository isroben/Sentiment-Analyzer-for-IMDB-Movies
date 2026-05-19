import numpy as np
import re

from nltk.corpus import stopwords
from nltk import PorterStremmer, sent_tokenize
from gensim.models import Word2Vec, KeyedVectors
from tqdm import tqdm
from gensim.utils import simple_preprocess

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


from src.utils.exception import CustomException
from src.utils.logger import get_logger



logger = get_logger(__name__)

class TextCleaner:
    """ Stateless text cleaning"""

    def __init__(self):
        self.sw_list = stopwords.words('english')
        self.ps = PorterStremmer()

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
    def __init__(self, window=10, min_count=2, vector_size=100, epochs=10):
        self.window = window
        self.min_count = min_count
        self.vector_size = vector_size
        self.epochs = epochs
        self.model = None

    
    def _build_sentences(self, texts):
        story = []
        for doc in texts:
            for sent in sent_tokenize(doc):
                story.append(simple_preprocess(sent))

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
            total_examples=self.model.coupus_count,
            epochs = self.epochs
        )

        return self
    
    def _document_vector(self, doc):
        tokens = [w for w in doc.split() if w in self.model.wv.index_to_key]
        if not tokens:
            return np.zeros(self.model.vector_size)
        return  np.mean(self.model.wv[tokens], axis=0)
    
    def transform(self, texts):
        return np.array([self._document_vector(doc) for doc in texts])
    
    def fit_transform(self, X_train_texts):
        self.fit(X_train_texts)
        return self.transform(X_train_texts)
    



def run_transformation(data):
    logger.info("Text transformation process started.")

    cleaner = TextCleaner()

    data['review'] = data['review'].apply(cleaner.clean)

    encoder = LabelEncoder()
    y = encoder.fit_transform(data['sentiment'])

    X_train, X_test, y_train, y_test = train_test_split(data['review'].values, y, test_size=0.2, random_state=42)

    vectorizer = Vectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    return X_train, X_test, y_train, y_test, vectorizer, encoder