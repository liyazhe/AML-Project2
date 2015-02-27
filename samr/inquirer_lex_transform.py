#  Explain and drop some links here
from collections import namedtuple, defaultdict
import csv
import os

from samr.transformations import StatelessTransform
from samr.settings import DATA_PATH


FIELDS = ("Entry, Source, Positiv, Negativ, Pstv, Affil, Ngtv, Hostile, Strong,"
          " Power, Weak, Submit, Active, Passive, Pleasur, Pain, Feel, Arousal,"
          " EMOT, Virtue, Vice, Ovrst, Undrst, Academ, Doctrin, Econ, Exch, "
          "ECON, Exprsv, Legal, Milit, Polit, POLIT, Relig, Role, COLL, Work, "
          "Ritual, SocRel, Race, Kin, MALE, Female, Nonadlt, HU, ANI, PLACE, "
          "Social, Region, Route, Aquatic, Land, Sky, Object, Tool, Food, "
          "Vehicle, BldgPt, ComnObj, NatObj, BodyPt, ComForm, COM, Say, Need, "
          "Goal, Try, Means, Persist, Complet, Fail, NatrPro, Begin, Vary, "
          "Increas, Decreas, Finish, Stay, Rise, Exert, Fetch, Travel, Fall, "
          "Think, Know, Causal, Ought, Perceiv, Compare, Eval, EVAL, Solve, "
          "Abs, ABS, Quality, Quan, NUMB, ORD, CARD, FREQ, DIST, Time, TIME, "
          "Space, POS, DIM, Rel, COLOR, Self, Our, You, Name, Yes, No, Negate, "
          "Intrj, IAV, DAV, SV, IPadj, IndAdj, PowGain, PowLoss, PowEnds, "
          "PowAren, PowCon, PowCoop, PowAuPt, PowPt, PowDoct, PowAuth, PowOth, "
          "PowTot, RcEthic, RcRelig, RcGain, RcLoss, RcEnds, RcTot, RspGain, "
          "RspLoss, RspOth, RspTot, AffGain, AffLoss, AffPt, AffOth, AffTot, "
          "WltPt, WltTran, WltOth, WltTot, WlbGain, WlbLoss, WlbPhys, WlbPsyc, "
          "WlbPt, WlbTot, EnlGain, EnlLoss, EnlEnds, EnlPt, EnlOth, EnlTot, "
          "SklAsth, SklPt, SklOth, SklTot, TrnGain, TrnLoss, TranLw, MeansLw, "
          "EndsLw, ArenaLw, PtLw, Nation, Anomie, NegAff, PosAff, SureLw, If, "
          "NotLw, TimeSpc, FormLw, Othtags, Defined")

InquirerLexEntry = namedtuple("InquirerLexEntry", FIELDS)
FIELDS = InquirerLexEntry._fields


class InquirerLexTransform(StatelessTransform):
    _corpus = []
    _use_fields = [FIELDS.index(x) for x in "Positiv Negativ IAV Strong Weak If EMOT Active Passive Undrst Ovrst Negate SV RspGain RspLoss EnlGain EnlLoss EnlEnds EnlPt SklAsth SureLw NotLw NegAff PosAff TrnGain TrnLoss".split()]

    def transform(self, X, y=None):
        """
        `X` is expected to be a list of `str` instances containing the phrases.
        Return value is a list of `str` containing different amounts of the
        words "Positiv_Positiv", "Negativ_Negativ", "IAV_IAV", "Strong_Strong"
        based on the sentiments given to the input words by the Hardvard
        Inquirer lexicon.
        """
        corpus = self._get_corpus()
        result = []
        for phrase in X:
            newphrase = []
            for word in phrase.split():
                if "NOT_" in word:
                    newphrase.extend(self._get_inverse(corpus.get(word[4:].lower(),[])))
                else:
                    newphrase.extend(corpus.get(word.lower(), []))
            result.append(" ".join(newphrase))
        return result

    def _get_inverse(self,tags):
        dic={"Positiv_Positiv": "Negativ_Negativ","Negativ_Negativ": "Positiv_Positiv","Strong_Strong":"Weak_Weak", "Weak_Weak": "Strong_Strong",
             "Active_Active":"Passive_Passive", "Passive_Passive": "Active_Active","Undrst_Undrst":"Ovrst_Ovrst","Ovrst_Ovrst":"Undrst_Undrst",
             "RspLoss_RspLoss":"RspGain_RspGain","RspGain_RspGain":"RspLoss_RspLoss","EnlGain_EnlGain":"EnlLoss_EnlLoss","EnlLoss_EnlLoss":"EnlGain_EnlGain",
             "SureLw_SureLw":"NotLw_NotLw","NotLw_NotLw":"SureLw_SureLw", "NegAff_NegAff":"PosAff_PosAff","PosAff_PosAff":"NegAff_NegAff",
             "TrnGain_TrnGain":"TrnLoss_TrnLoss", "TrnLoss_TrnLoss":"TrnGain_TrnGain"}
        rtn=[]
        for tag in tags:
            if tag in dic.keys():
                rtn.append(dic[tag])
            else:
                rtn.append(tag)
        return rtn

    def _get_corpus(self):
        """
        Private method used to cache a dictionary with the Harvard Inquirer
        corpus.
        """
        if not self._corpus:
            corpus = defaultdict(list)
            it = csv.reader(open(os.path.join(DATA_PATH, "inquirerbasicttabsclean")),
                            delimiter="\t")
            next(it)  # Drop header row
            for row in it:
                entry = InquirerLexEntry(*row)
                xs = []
                for i in self._use_fields:
                    name, x = FIELDS[i], entry[i]
                    if x:
                        xs.append("{}_{}".format(name, x))
                name = entry.Entry.lower()
                if "#" in name:
                    name = name[:name.index("#")]
                corpus[name].extend(xs)
            self._corpus.append(dict(corpus))
        return self._corpus[0]

class StructuredInquirerLexTransform(StatelessTransform):
    _corpus = []
    _use_fields = [FIELDS.index(x) for x in "Positiv Negativ IAV Strong Pstv Ngtv Weak".split()]

    def __init__(self,split_words):
        self.split_pattern='|'.join(split_words)

    def transform(self, X, y=None):
        """
        `X` is expected to be a list of `str` instances containing the phrases.
        Return value is a list of `str` containing different amounts of the
        words "Positiv_Positiv", "Negativ_Negativ", "IAV_IAV", "Strong_Strong"
        based on the sentiments given to the input words by the Hardvard
        Inquirer lexicon.
        """
        import re
        corpus = self._get_corpus()
        result = []
        for phrase in X:
            subphrases=re.split(self.split_pattern,phrase)
            for subphrase in subphrases:
                newphrase = []
                for word in subphrase.split():
                    tag=corpus.get(word.lower(), [])
                    if not tag in newphrase:
                        newphrase.extend(corpus.get(word.lower(), []))
                newsentence="-".join(newphrase)
            result.append(newsentence)
        return result

    def _get_corpus(self):
        """
        Private method used to cache a dictionary with the Harvard Inquirer
        corpus.
        """
        if not self._corpus:
            corpus = defaultdict(list)
            it = csv.reader(open(os.path.join(DATA_PATH, "inquirerbasicttabsclean")),
                            delimiter="\t")
            next(it)  # Drop header row
            for row in it:
                entry = InquirerLexEntry(*row)
                xs = []
                for i in self._use_fields:
                    name, x = FIELDS[i], entry[i]
                    if x:
                        xs.append("{}_{}".format(name, x))
                name = entry.Entry.lower()
                if "#" in name:
                    name = name[:name.index("#")]
                corpus[name].extend(xs)
            self._corpus.append(dict(corpus))
        return self._corpus[0]