import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from unidecode import unidecode


def remove_urls(text):
    temp = re.sub("http.?://[^\s]+[\s]?", "", text)
    text = re.sub("(www\.)[^\s]+[\s]?", "", temp)
    return text


def remove_digits(text: str):
    return re.sub(r'\d+', '', text)


def clean_special_chars(text: str):
    # return re.sub('[^A-Za-z0-9 $]+', '', text)
    return re.sub('[^A-Za-z0-9 ]+', '', text)


def preprocess(text: str):
    return remove_urls(remove_digits(clean_special_chars(unidecode(text.lower().capitalize()))))
