from nltk.stem import RSLPStemmer
import re
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


def to_lower(input_text):
    return input_text.lower()

def removerCaracteresEspeciais(input_text):
        input_text = re.sub(u'[áãâà]', 'a', input_text)
        input_text = re.sub(u'[éèê]', 'e', input_text)
        input_text = re.sub(u'[íì]', 'i', input_text)
        input_text = re.sub(u'[óõôò]', 'o', input_text)
        input_text = re.sub(u'[úùü]', 'u', input_text)
        input_text = re.sub(u'[ç]', 'c', input_text)
        return input_text

def remove_urls(input_text):
    temp = re.sub("http.?://[^\s]+[\s]?", "",input_text)
    input_text = re.sub("(www\.)[^\s]+[\s]?", "",temp)
    return input_text

def remove_punctuation(input_text):
    # Make translation table
    punct = string.punctuation
    trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
    return input_text.translate(trantab)

def remove_digits(input_text):
    import re
    return re.sub('\d+', '', input_text)

def remove_stopwords(input_text, stopwords_list):
    whitelist = [""]
    words = input_text.split() 
    clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
    return " ".join(clean_words)

def lemmatization(input_text):
    lemmatizer = WordNetLemmatizer()
    words = input_text.split() 
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(lemmatized_words)

def stemming(input_text):
    porter = PorterStemmer()
    words = input_text.split() 
    stemmed_words = [porter.stem(word) for word in words]
    return " ".join(stemmed_words)

def rslps(input_text):
    st = RSLPStemmer()
    words = input_text.split()
    stemmed_words = [st.stem(word) for word in words]
    return " ".join(stemmed_words)

def tratarnomecolunas(input_text):
    input_text = re.sub("(\(eof\))|\(|\)","",input_text)
    input_text = re.sub("(_codigo/nome)|(_cod/nome)|(_dia/mes/ano)|","",input_text)
    return input_text
    
# Trata casos especificos identificados após análise dos textos
def remocaoProfunda(input_text):
    temp = input_text.replace("\t"," ")
    temp = re.sub("(\n\s{1})|(\s{1}\n)","\n",temp)
    temp = re.sub("\s{2,}", " ",temp)
    temp = temp.replace("-\n", "")
    temp = re.sub("(\d+)x(\d+|\W|\s)", "",temp)
    temp = re.sub("(r\$)|[¹²³ªº°¿øµ]", "",temp)
    input_text = re.sub("n(\.){0,1}[º|°]{0,1}\s{0,1}\d", "",temp)
    return input_text

# Faz todo o tratamento de texto (retira pontuação, stopwords, números)
def cleanTextData(texto):
    # Faz o download das stopwords
    try:
        stopwords_list = stopwords.words('portuguese')
    except:
        nltk.download("stopwords")
        nltk.download('wordnet')
        stopwords_list = stopwords.words('portuguese')
    
    stopwords_list.append("pdf")
    stopwords_list.append("total")
    stopwords_list.append("mes")
    stopwords_list.append("goias")
    stopwords_list.append("go")
    result = [0]*len(texto)
    temp = None
    for i in range(len(texto)):
        temp = to_lower(texto.iloc[i])
        temp = removerCaracteresEspeciais(temp)
        temp = remocaoProfunda(temp)
        temp = remove_urls(temp)
        temp = remove_punctuation(temp)
        temp = remove_digits(temp)
        temp = remove_stopwords(temp, stopwords_list)
        result[i] = temp
#        result.append(stemming(temp))
#        result.append(rslps(temp))
    return result

# Calcula o TFIDF
def calculaTFIDF(textoTratado):
    cv = TfidfVectorizer(dtype=np.float32)
    data_cv = cv.fit_transform(textoTratado)
    tfidf = pd.DataFrame.sparse.from_spmatrix(data_cv, columns = cv.get_feature_names())
#    tfidf = pd.DataFrame(data_cv.toarray(), columns = cv.get_feature_names())
    return tfidf