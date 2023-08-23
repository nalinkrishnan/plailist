import re
import json
import openai
from lyricsgenius import Genius
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from top2vec import Top2Vec

docs = []
titles_artists = []

genius_access_token = 'wjybx3D8JdeyaQkkoezVikd3B2YhexZHEiJRcFXi6pVafExahHJmvEN2PioP36F8'
genius_secret_key = b'FUB9WH4hUIiLxFFgofyaPuQVjLYPJ1YARk-CFfHHkAYA5EKFUsaK7EVzJGdeTEdY3dcfi0zemgv5_C5IGWM7Qw'
openai.api_key = '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'

f = open('Apple Music Library Tracks.json')
data = json.load(f)

for track in data:
  title = track['Title']
  artist = track['Artist']
  feat_index = title.find("(feat.")
  with_index = title.find("(with")

  if feat_index != -1:
    title = title[:feat_index]
  elif with_index != -1:
    title = title[:with_index]
  
  title = title.strip()
  genius = Genius(genius_access_token)
  song = genius.search_song(title, track['Artist'])
  lyrics = []

  if song == None:
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "user", "content": f"Give me a list of words and phrases that describe the theme of the song \"{title}\" by {track['Artist']}."}
      ]
    )

    lines = completion.choices[0].message.content.split('\n')
    for line in lines:
        parts = line.split()
        words = parts[1:]
        for word in words:
          word = word.lower()
          word = word.rstrip('.')
          lyrics.append(word)
    
    doc = ' '.join(lyrics)
    docs.append(doc)
      
  else:
    lyrics = song.lyrics.split()

      # download stopwords and punkt if not already downloaded
    nltk.download('stopwords')
    nltk.download('punkt')

      # define regex patterns
    num_pattern = r'\d+'
    name_pattern = r'[A-Z][a-z]+(\s[A-Z][a-z]+)*'
    bracket_pattern = r'\[[^\]]*\]'
    contraction_pattern = r"\b\w+(?:'\w+)?"

    # load stopwords
    stop_words = set(stopwords.words('english'))

    # tokenize lyrics using RegexpTokenizer
    tokenizer = RegexpTokenizer(contraction_pattern)
    lyrics = tokenizer.tokenize(song.lyrics)

    # filter out stopwords, non-English words, numbers, names, and characters between square brackets
    filtered_lyrics = [word.lower() for word in lyrics if word.lower() not in stop_words and re.match(r'^[a-zA-Z]+$', word) and not re.match(num_pattern, word) and not re.match(name_pattern, word) and not re.match(bracket_pattern, word)]
    # filtered_lyrics = [word.lower() for word in lyrics if word.lower() not in stop_words]
    # stemmer = PorterStemmer()
    # stemmed_lyrics = [stemmer.stem(token) for token in filtered_lyrics]
    # lemmatizer = WordNetLemmatizer()
    # lemmatized_lyrics = []
    # for token in stemmed_lyrics:
    #     tag = nltk.pos_tag([token])[0][1][0].upper()
    #     tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    #     pos = tag_dict.get(tag, wordnet.NOUN)
    #     lemma = lemmatizer.lemmatize(token, pos=pos)
    #     lemmatized_lyrics.append(lemma)
    doc = ' '.join(filtered_lyrics)
    docs.append(doc)

  titles_artists.append((track['Title'], track['Artist']))
  # if len(docs) == 3:
  #   print(docs)
  #   print(titles_artists)
  #   breakpoint()


model = Top2Vec(documents=docs, speed="fast-learn", workers=8)


num_topics = model.get_num_topics()
print("Number of topics: ", num_topics)
topic_words, word_scores, topic_nums = model.get_topics(num_topics)
print("Total Topic Words: ", topic_words)
print("Total Word Scores: ", word_scores)
print("Total Topic Index: ", topic_nums)

topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["destiny"], num_topics=5)

print("Topic Words: ", topic_words)
print("Word Scores: ", word_scores)
print("Topic Scores: ", topic_words)
print("Topic Index: ", topic_nums)

for num in topic_nums:
  documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=48, num_docs=5)
  for doc, score, doc_id in zip(documents, document_scores, document_ids):
      print(f"Document: {doc_id}, Title: {titles_artists[doc_id][0]}, Artist: {titles_artists[doc_id][1]}, Score: {score}")
      print("-----------")
      print(doc)
      print("-----------")
      print()





