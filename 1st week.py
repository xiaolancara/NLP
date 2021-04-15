import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import gutenberg, wordnet

example_text = "Hello Mr. Smith, how are you doing today? The weather is great and python us awesome. The sky is pinkish-blue. You should not eat cardboard."
# print(sent_tokenize(example_text)) # split to sentences
# print(word_tokenize(example_text)) # split to words

"""
Stop Words
"""
example_sentence = "This is an example showing off stop word filteration."
stop_words = set(stopwords.words("english"))
# print(stop_words)
words = word_tokenize(example_sentence)
filtered_sentence = [w for w in words if not w in stop_words]
#print(filtered_sentence)

"""
Stemmming
"""
ps = PorterStemmer()
example_words = ["Python","Pythoner","pythoning","pythoned","pythonly"]

new_text = "It is very important for pythonly while you are pythoning with python. All Pythoners have pythoned poorly at least once."

words = word_tokenize(new_text)
stem_words = [ps.stem(w) for w in words]
# print(stem_words)

"""
Part of Speech Tagging
"""
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
tokenized = custom_sent_tokenizer.tokenize(sample_text)

"""
Chunking
"""
"""
Chinking
"""
def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            #chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?} """
            chunkGram = r"""Chunk: {<.*>+} 
                                       }<VB.?|IN|DT>+{"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()
            #print(chunked)
            #print(tagged)
    except Exception as e:
        print(str(e))

"""
Named Entity
"""
def process_content():
    try:
        for i in tokenized[5:15]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            namedEnt = nltk.ne_chunk(tagged, binary=True) # Named Entity Recognition, binary=True will combine all entities nearby to one
            #namedEnt.draw()
    except Exception as e:
        print(str(e))
#process_content()

"""
Lemmatizing
"""
lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize("better",pos="a"))
# print(lemmatizer.lemmatize("best",pos="a"))
# print(lemmatizer.lemmatize("run"))
# print(lemmatizer.lemmatize("run",'v'))

"""
NLTK Corpora
"""
sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)
#print(tok[5:15])

"""
WordNet
"""
syns = wordnet.synsets("program")

# synset
print(syns[0].name())
# just the word
print(syns[0].lemmas()[0].name())

# definition
print(syns[0].definition())

# examples
print(syns[0].examples())

synonyms = [] # similar words
antonyms = [] # opposite words

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        # print(l)
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
print(set(synonyms))
print(set(antonyms))

"""
semantic similarity
"""
# rewrite things
# or reverse the similarity to see if they're people cheating on papers by switching words
word1 = wordnet.synset("ship.n.01") # noun
word2 = wordnet.synset("boat.n.01") # noun
print(word1.wup_similarity(word2))

word1 = wordnet.synset("ship.n.01") # noun
word2 = wordnet.synset("car.n.01") # noun
print(word1.wup_similarity(word2))

word1 = wordnet.synset("ship.n.01") # noun
word2 = wordnet.synset("cactus.n.01") # noun
print(word1.wup_similarity(word2))


