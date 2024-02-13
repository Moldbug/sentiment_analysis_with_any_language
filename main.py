import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from googletrans import Translator

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('wordnet')

text_input = input()


def googletranslator(t):
    translator = Translator()
    return translator.translate(t)


textTR = googletranslator(text_input)
# print(textTR.text)


lemma = WordNetLemmatizer()
word_tokens = word_tokenize(textTR.text)
sentiment = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

lemmatized_text = ' '.join([lemma.lemmatize(word) for word in word_tokens])
print(lemmatized_text)

def returnsentiment(s):
    score = sentiment.polarity_scores(s)['compound']

    if score > 0:
        sent = 'Positive'
    elif score == 0:
        sent = 'Neutral'
    else:
        sent = 'Negative'
    return score, sent


if __name__ == "__main__":
    print(returnsentiment(lemmatized_text))
