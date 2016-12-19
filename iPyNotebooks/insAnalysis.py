# -*- coding: utf-8 -*-
import pandas as pd
import nltk, re, pprint
# coding=UTF-8


test = "every time i call i get a different explanation of why something happened. even the notes that you have two people can't tell me the same story."
grammar = """
    NP: {<DT>?<JJ>*<NN>}
        {<DT|PP\$>?<JJ>*<NN>}  # chunk determiner/possessive, adjectives and noun
        {<NNP>+} # chunk sequences of proper nouns
        {<NN><NN>}  # Chunk two consecutive nouns
"""
cp = nltk.RegexpParser(grammar)
def ie_preprocess(document):
    sentences = nltk.sent_tokenize(document)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [nltk.pos_tag(sent) for sent in sentences]
    return(sentences)
def tags_since_dt(sentence, i):
     tags = set()
     for word, pos in sentence[:i]:
         if pos == 'DT':
             tags = set()
         else:
             tags.add(pos)
     return '+'.join(sorted(tags))
def npchunk_features(sentence, i, history):
     word, pos = sentence[i]
     if i == 0:
         prevword, prevpos = "<START>", "<START>"
     else:
         prevword, prevpos = sentence[i-1]
     if i == len(sentence)-1:
         nextword, nextpos = "<END>", "<END>"
     else:
         nextword, nextpos = sentence[i+1]
     return {"pos": pos,
             "word": word,
             "prevpos": prevpos,
             "nextpos": nextpos,
             "prevpos+pos": "%s+%s" % (prevpos, pos), 
             "pos+nextpos": "%s+%s" % (pos, nextpos),
             "tags-since-dt": tags_since_dt(sentence, i)} 


from nltk.corpus import conll2000
train2 = conll2000.chunked_sents('test.txt',chunk_types=['NP'])

chunker = ConsecutiveNPChunker(train2)
grammar2 = r"NP: {<[CDJNP].*>+}"

cp2 = nltk.RegexpParser(grammar2)
test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
print(chunker.evaluate(test_sents)) 
import pickle
#pickle.dump(chunker,open('chunk2','wb'))
with open('chunk2', 'r') as f:
    chunker = pickle.load(f)
out = ie_preprocess(test)
x = chunker.parse(out[1])

lemmatizer = nltk.WordNetLemmatizer()
stemmer = nltk.stem.porter.PorterStemmer()

from nltk.corpus import stopwords
stopwords = stopwords.words('english')
#code to normalize, clean and extract key words from noun phrases
def leaves(tree):
    """Finds NP (nounphrase) leaf nodes of a chunk tree."""
    for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
        yield subtree.leaves()
def normalise(word):
    """Normalises words to lowercase and stems and lemmatizes it."""
    word = word.lower()
    word = stemmer.stem_word(word)
    word = lemmatizer.lemmatize(word)
    return word

def acceptable_word(word):
    """Checks conditions for acceptable word: length, stopword."""
    accepted = bool(2 <= len(word) <= 40
        and word.lower() not in stopwords)
    return accepted


def get_terms(tree):
    for leaf in leaves(tree):
        term = [ normalise(w) for w,t in leaf if acceptable_word(w) ]
        yield term

def keywords(chunked):
    terms = get_terms(chunked)
    
    nouns = []
    for term in terms:
        wrd = ""
        for word in term:
            if wrd == "":
                wrd = word
            else:
                wrd+=" " + word
        if wrd <> "":
            nouns.append(wrd)
    return(nouns)
            
df = pd.read_csv('C:\Users\jswortz\Documents\UHG\\allsurveys_201512.csv') #import data

df = df[pd.notnull(df['feedback'])] # drop null rows of data

z = []
for sent in df.feedback:
    sent = sent.replace("'","%")
    sent = ie_preprocess(sent)
    out = []
    for s in sent:
        chnk = chunker.parse(s)
        chnk2 = keywords(chnk)
        out+=chnk2
    z.append(out)
      

zz = zip(z,df.Adj_Sentiment_Score,df.complete_date,df.agentname)
agent=unique(df.agentname) #find unique dates and agents to iterate over

dates = unique(df.complete_date)

sentiment = pd.DataFrame() #declare empty data frame to appeand against at the end of the iterations
wordcount={}
for d in dates:
    for ag in agent:
        wordcount = {}
        for line in zz:
            if line[3] == ag and line[2] == d:
                for word in line[0]:
                    if word not in wordcount:
                        wordcount[word] = [line[1]] #set the first element of the list to be the sentiment
                        wordcount[word].append(1) #append the second element of the list to be the first counter
                    else:
                        wordcount[word][0] += line[1] #increment sentiment score 
                        wordcount[word][1] += 1 #increment count

        st = pd.DataFrame(wordcount)
        st = st.transpose()
        #st.columns = ["Sentiment Sum","Count"]
        st['name'] = ag #set agent name in the data
        st['date'] = d # set date in data
        sentiment = sentiment.append(st) #append date-agent chunk to master data frame

sentiment.columns = ['Sentiment Sum','Count','Date','Agent']
sentiment.to_csv(path_or_buf='c:\\users\\jswortz\\nltkoutput2.csv')



class ConsecutiveNPChunkTagger(nltk.TaggerI): 

    def __init__(self, train_sents):
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append( (featureset, tag) )
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(
            train_set, trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI): 
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

