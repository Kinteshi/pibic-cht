import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
#from unidecode import unidecode


def tolower(text: str):
    return text.lower()


def remove_urls(text):
    temp = re.sub("http.?://[^\s]+[\s]?", "", text)
    text = re.sub("(www\.)[^\s]+[\s]?", "", temp)
    return text


def remove_digits(text: str):
    return re.sub(r'\d+', '', text)


def remove_punctuation(text: str):
    punct = string.punctuation
    translation_table = str.maketrans(punct, len(punct)*' ')
    return text.translate(translation_table)


def remove_stopwords(text: str):
    stopwords_list = stopwords.words('portuguese')
    return ' '.join([word for word in text.split(' ') if (word not in stopwords_list) and len(word) > 1])


def lemmatize(text: str):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split(' ')])


def clean_special_chars(text: str):
    return re.sub('[^A-Za-z0-9 $]+', '', text)


def stemming(text: str):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in text.split(' ')])


def tfidfPprocessing(text):
    return stemming(remove_stopwords(remove_punctuation(remove_digits(remove_urls(tolower(clean_special_chars(text)))))))


def embeddingPrep(text: str):
    return remove_urls(remove_digits(clean_special_chars(text.lower().capitalize())))
