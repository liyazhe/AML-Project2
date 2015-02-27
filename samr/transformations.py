"""
This module implements several scikit-learn compatible transformers, see
scikit-learn documentation for the convension fit/transform convensions.
"""

import numpy
import re
import string

from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import fit_ovo
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import sentiwordnet as swn

class StatelessTransform:
    """
    Base class for all transformations that do not depend on training (ie, are
    stateless).
    """
    def fit(self, X, y=None):
        return self

class ExtractLemma(StatelessTransform):
    def transform(self,X):
        porter_stemmer = PorterStemmer()
        it = (" ".join(map(porter_stemmer.stem,nltk.word_tokenize(x))) for x in X)
        return [x.lower() for x in it]

class ExtractText(StatelessTransform):
    """
    This should be the first transformation on a samr pipeline, it extracts
    the phrase text from the richer `Datapoint` class.
    """
    def __init__(self, lowercase=False):
        self.lowercase = lowercase

    def transform(self, X):
        """
        `X` is expected to be a list of `Datapoint` instances.
        Return value is a list of `str` instances in which words were tokenized
        and are separated by a single space " ". Optionally words are also
        lowercased depending on the argument given at __init__.
        """
        it = (" ".join(nltk.word_tokenize(datapoint.phrase)) for datapoint in X)
        if self.lowercase:
            return [x.lower() for x in it]
        return list(it)

class ReplaceText(StatelessTransform):
    def __init__(self, replacements):
        """
        Replacements should be a list of `(from, to)` tuples of strings.
        """
        self.rdict = dict(replacements)
        self.pat = re.compile("|".join(re.escape(origin) for origin, _ in replacements))

    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        Return value is also a list of `str` instances with the replacements
        applied.
        """
        if not self.rdict:
            return X
        return [self.pat.sub(self._repl_fun, x) for x in X]

    def _repl_fun(self, match):
        return self.rdict[match.group()]


class MapToSynsets(StatelessTransform):
    """
    This transformation replaces words in the input with their Wordnet
    synsets[0].
    The intuition behind it is that phrases represented by synset vectors
    should be "closer" to one another (not suffer the curse of dimensionality)
    than the sparser (often poetical) words used for the reviews.

    [0] For example "bank": http://wordnetweb.princeton.edu/perl/webwn?s=bank
    """
    def transform(self, X):
        """
        `X` is expected to be a list of `str` instances.
        It returns a list of `str` instances such that the i-th element
        containins the names of the synsets of all the words in `X[i]`,
        excluding noun synsets.
        `X[i]` is internally tokenized using `str.split`, so it should be
        formatted accordingly.
        """
        return [self._text_to_synsets(x) for x in X]

    def _text_to_synsets(self, text):
        result = []
        for word in text.split():
            if "NOT_" in word:
                ss = nltk.wordnet.wordnet.synsets(word[4:])
                for syn in ss:
                    lms=syn.lemmas()
                    for lm in lms:
                        result.extend([str(x.synset()) for x in lm.antonyms()])
            else:
                ss = nltk.wordnet.wordnet.synsets(word)
                result.extend(str(s) for s in ss) #if ".n." not in str(s)
        return " ".join(result)

class MapToSenti(StatelessTransform):
    def transform(self,X):
        return [self._text_to_senti(x) for x in X]

    def _text_to_senti(self,sentence):
        i=0.0
        score_pos=0.0
        score_neg=0.0
        #tags=nltk.pos_tag(sentence.split())
        for token in sentence.split():
            i+=1.0
            #if 'RB' in tags[int(i-1)]:
            #    adjs=self._find_adj(token)
            #    if len(adjs)!=0:
            #        token=adjs[0]
            scores=self._find_scores(token)
            score_pos+=scores[0]
            score_neg+=scores[1]
        if i==0.0:
            i=1.0
        return [score_pos/i,score_neg/i]

    def _find_scores(self,token):
        max_pos=0.0
        max_neg=0.0
        for t in swn.senti_synsets(token):
            #t=swn.senti_synset(str(sent))
            pos=t.pos_score()
            neg=t.neg_score()
            if pos > max_pos:
                max_pos = pos
            if neg > max_neg:
                max_neg = neg
        return [max_pos, max_neg]

    def _find_adj(self,word):
        bag=nltk.wordnet.wordnet.synsets(word)
        adjsense=[]
        for sense in bag:
                if ".r." in str(sense):
                    for lemma in sense.lemmas():
                        for x in lemma.pertainyms():
                            if (".s." in str(x)) or (".a." in str(x)):
                                adjsense.append(str(x.name()))
        return adjsense

class MapToOsgoodScores(StatelessTransform):
    good=nltk.wordnet.wordnet.synset('good.a.01')
    bad=nltk.wordnet.wordnet.synset('bad.a.01')
    strong=nltk.wordnet.wordnet.synset('strong.a.01')
    weak=nltk.wordnet.wordnet.synset('weak.a.01')
    active=nltk.wordnet.wordnet.synset('active.a.03')
    passive=nltk.wordnet.wordnet.synset('passive.a.01')
    def __init__(self,stopwords):
        self.stopwords=stopwords

    def transform(self,X):
        return [self._text_to_senti_score(x) for x in X]

    def _text_to_senti_score(self,sentence):
        score_good=0.0
        score_bad=0.0
        score_strong=0.0
        score_weak=0.0
        score_active=0.0
        score_passive=0.0

        i=0.0
        for token in sentence.split():
            i+=1.0
            score_good+=self._dist(token,self.good)
            score_bad+=self._dist(token,self.bad)
            score_strong+=self._dist(token,self.strong)
            score_weak+=self._dist(token,self.weak)
            score_active+=self._dist(token,self.active)
            score_passive+=self._dist(token,self.passive)

        return [score_good/i,score_bad/i, score_strong/i,score_weak/i, score_active/i,score_passive/i]

    def _dist(self,token,base):
        max=0.0
        bag=nltk.wordnet.wordnet.synsets(token)
        if len(bag)!=0:
            for sense in bag:
                if ".r." in str(sense):
                    adjsense=[]
                    for lemma in sense.lemmas():
                        for x in lemma.pertainyms():
                            adjsense.append(x.synset())
                    if len(adjsense)!=0:
                        sense=adjsense
                    else:
                        sense=[sense]
                else:
                    sense=[sense]
                for s in sense:
                    score=base.path_similarity(s)
                    if not score is None and max < score:
                        max=score
        else:
            max=0.0
        return max

class MapToPartOfSpeech(StatelessTransform):
    def transform(self,X):
        return [self._to_part_of_speech(x) for x in X]

    def _to_part_of_speech(self,x):
        pos=[tag for (word,tag) in nltk.pos_tag(x.split())]
        return ' '.join(pos)

class ReplaceNegation(StatelessTransform):
    def __init__(self, negations,sentenceStarter):
        self.negations=negations
        self.sentenceStarter=sentenceStarter

    def transform(self,X):
        return [self._transform_negation(x) for x in X]

    def _transform_negation(self,x):
        rtn=[]
        flag=False
        for token in x.split():
            if flag:
                token = "NOT_"+token

            if token in self.negations:
                flag=not flag

            if token in self.sentenceStarter:
                flag=False

            rtn.append(token)
        return ' '.join(rtn)

class Densifier(StatelessTransform):
    """
    A transformation that densifies an scipy sparse matrix into a numpy ndarray
    """
    def transform(self, X, y=None):
        """
        `X` is expected to be a scipy sparse matrix.
        It returns `X` in a (dense) numpy ndarray.
        """
        return X.todense()


class ClassifierOvOAsFeatures:
    """
    A transformation that esentially implement a form of dimensionality
    reduction.
    This class uses a fast SGDClassifier configured like a linear SVM to produce
    a vector of decision functions separating target classes in a
    one-versus-rest fashion.
    It's useful to reduce the dimension bag-of-words feature-set into features
    that are richer in information.
    """
    def fit(self, X, y):
        """
        `X` is expected to be an array-like or a sparse matrix.
        `y` is expected to be an array-like containing the classes to learn.
        """
        self.classifiers = fit_ovo(SGDClassifier(), X, numpy.array(y), n_jobs=-1)[0]
        return self

    def transform(self, X, y=None):
        """
        `X` is expected to be an array-like or a sparse matrix.
        It returns a dense matrix of shape (n_samples, m_features) where
            m_features = (n_classes * (n_classes - 1)) / 2
        """
        xs = [clf.decision_function(X).reshape(-1, 1) for clf in self.classifiers]
        return numpy.hstack(xs)
