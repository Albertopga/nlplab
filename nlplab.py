
import bs4 as bs
import urllib.request
import json, requests
import sys
import re
import heapq
import ssl
import nltk

from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud


#nltk.download('punkt')
#nltk.download('stopwords')

SENTENCES = 16


def parse_article(article):
    parsed_article = bs.BeautifulSoup(article, "html.parser")

    paragraphs = parsed_article.find_all('p')

    article_text = ""

    for p in paragraphs:
        article_text += p.text

    # Removing Square Brackets and Extra Spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)

    # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

    sentence_list = nltk.sent_tokenize(article_text)

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(formatted_article_text):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(SENTENCES, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    return summary


def wordcloud(article):
    # Create stopword list:
    #stopwords = set(STOPWORDS)
    #stopwords.update(["br", "href"])
    #wordcloud = WordCloud(stopwords=stopwords).generate(article)
    wordcloud = WordCloud().generate(article)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig('wordcloud11.png')
    plt.show()



url = sys.argv[1]

story_data_request = urllib.request.Request(url)
response = urllib.request.urlopen(story_data_request)
summary = parse_article(response)

wordcloud(summary)

print(url)
print("Summary:")
print(summary)
