# -*- coding: utf-8 -*-
# Copyright 2016 Sabino Miranda-Jiménez and Daniela Moctezuma
# with collaborations of Eric S. Tellez

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
from pyfreeling import Analyzer
from bs4 import BeautifulSoup
from lxml import etree

import io
import re
import os
import logging
from nltk.stem.snowball import SnowballStemmer
from b4msa.params import OPTION_NONE
from nltk.stem.porter import PorterStemmer
idModule = "language_dependency"
logger = logging.getLogger(idModule)
ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# formatterC = logging.Formatter('%(asctime)s\t%(levelname)s\t%(filename)s\t%(message)s')
formatterC = logging.Formatter('%(module)s-%(funcName)s\n\t%(levelname)s\t%(message)s')
ch.setFormatter(formatterC)
logger.addHandler(ch)

PATH = os.path.join(os.path.dirname(__file__), 'resources')


_HASHTAG = '#'
_USERTAG = '@'
_sURL_TAG = '_url'
_sUSER_TAG = '_usr'
_sHASH_TAG = '_htag'
_sNUM_TAG = '_num'
_sDATE_TAG = '_date'
_sENTITY_TAG = '_ent'
_sNEGATIVE = "_neg"
_sPOSITIVE = "_pos"
_sNEUTRAL = "_neu"


class LangDependencyError(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return repr(self.message)


class LangDependency():
    """
    Defines a set of functions to change text using laguage dependent transformations, e.g., 
    - Negation
    - Stemming
    - Stopwords
    """
    STOPWORDS_CACHE = {}
    NEG_STOPWORDS_CACHE = {}
    DICTIONARY_WORDS_CACHE = {}
    ABBREVIATION_WORDS_CACHE = {}

    def __init__(self, lang="spanish"):
        """
        Initializes the parameters for specific language
        """
        # DOUGLAS - Included "brazilian_portuguese" language in self.languages
        self.languages = ["spanish", "english", "italian", "german", "portuguese"]
        self.lang = lang
        self.correction = True
        self.lem = True
        self.del_ent = True

        if self.lang not in self.languages:
            raise LangDependencyError("Language not supported: " + lang)
        
        # DOUGLAS - TODO Implement negstopwords and negation for Brazilian Portuguese
        self.stopwords = LangDependency.STOPWORDS_CACHE.get(lang, None)
        if self.stopwords is None:
            self.stopwords = self.load_stopwords(os.path.join(PATH, "{0}.stopwords".format(lang)))
            LangDependency.STOPWORDS_CACHE[lang] = self.stopwords

        self.neg_stopwords = LangDependency.NEG_STOPWORDS_CACHE.get(lang, None)
        if self.neg_stopwords is None:
            self.neg_stopwords = self.load_stopwords(os.path.join(PATH, "{0}.neg.stopwords".format(lang)))
            LangDependency.NEG_STOPWORDS_CACHE[lang] = self.neg_stopwords

        if self.lang not in SnowballStemmer.languages:
            raise LangDependencyError("Language not supported for stemming: " + lang)
        if self.lang == "english":
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = SnowballStemmer(self.lang)

        self.dictionary_words = LangDependency.DICTIONARY_WORDS_CACHE.get(lang, None)
        if self.dictionary_words is None:
            # DOUGLAS - Although not being stopwords, the same method for loading stopwords is applicable.
            self.dictionary_words = self.load_dictionary(os.path.join(PATH, "{0}.dictionary".format(lang)))
            LangDependency.DICTIONARY_WORDS_CACHE[lang] = self.dictionary_words

        self.abbreviation_words = LangDependency.ABBREVIATION_WORDS_CACHE.get(lang, None)
        if self.abbreviation_words is None:
            # DOUGLAS - Load abbreviations from a file
            self.abbreviation_words = self.load_abbreviations(os.path.join(PATH, "{0}.abbreviations".format(lang)))
            LangDependency.ABBREVIATION_WORDS_CACHE[lang] = self.abbreviation_words

    def load_stopwords(self, fileName):
        """
        it loads stopwords from file
        """
        logger.debug("loading stopwords... " + fileName)
        if not os.path.isfile(fileName):
            raise LangDependencyError("File not found: " + fileName)                            
        
        StopWords = []
        with io.open(fileName, encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip().lower()
                if line == "":
                    continue
                if line.startswith("#"):
                    continue
                StopWords.append(line)

        return StopWords

    def load_dictionary(self, fileName):
        """
        it loads stopwords from file
        """
        logger.debug("loading dictionary... " + fileName)
        if not os.path.isfile(fileName):
            raise LangDependencyError("File not found: " + fileName)                            
        
        Dictionary = []
        with io.open(fileName, encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip().lower()
                if line == "":
                    continue
                if line.startswith("#"):
                    continue
                if re.match("[a-z]{1}", line, re.IGNORECASE):
                    continue
                Dictionary.append(line)

        return Dictionary

    def load_abbreviations(self, fileName):
        """
        it loads abbreviations from file
        """
        logger.debug("loading abbreviations... " + fileName)
        if not os.path.isfile(fileName):
            raise LangDependencyError("File not found: " + fileName)                             
        
        Abbreviations = {}
        with io.open(fileName, encoding='utf8') as f:
            for line in f.readlines():
                line = line.strip().lower()
                if line == "":
                    continue
                if line.startswith("#"):
                    continue

                abb, exp = line.split("\t")
                abb = abb.decode("utf-8")
                Abbreviations[abb] = exp

        return Abbreviations
                
    def stemming(self, text):
        """
        Applies the stemming process to `text` parameter
        """
        
        tokens = re.split(r"~", text.strip())
        t = []
        for tok in tokens:
            if re.search(r"^(@|#|_|~)", tok, flags=re.I):
                t.append(tok)
            else:
                t.append(self.stemmer.stem(tok))
        return "~".join(t)

    # DOUGLAS - Lemmatizing for portuguese with freeling. Extract only lemmas from Freeling response
    def lemmatizing(self, text):
        """
        Applies lemmatizing process to the given text
        """
        if self.lang not in self.languages:
            raise LangDependencyError("Lemmatizing - language not defined")
        
        if self.lang == "portuguese":
            text = self.portuguese_lemmatizing(text)
        elif self.lang == "spanish":
            raise LangDependencyError("Lemmatizing - language not implemented for lemmatizing")
        elif self.lang == "english":
            raise LangDependencyError("Lemmatizing - language not implemented for lemmatizing")
        elif self.lang == "italian":
            raise LangDependencyError("Lemmatizing - language not implemented for lemmatizing")

        return text

    def negation(self, text):
        """
        Applies negation process to the given text
        """
        if self.lang not in self.languages:
            raise LangDependencyError("Negation - language not defined")
        
        if self.lang == "spanish":
            text = self.spanish_negation(text)
        elif self.lang == "english":
            text = self.english_negation(text)
        elif self.lang == "italian":
            text = self.italian_negation(text)
        elif self.lang == "portuguese":
            text = self.portuguese_negation(text)

        return text

    def error_correction(self, text):
        """
        Applies error correction process to the given text
        """
        if self.lang not in self.languages:
            raise LangDependencyError("Error Correction - language not defined")
        
        if self.lang == "portuguese":
            text = self.portuguese_correction(text)
        elif self.lang == "spanish":
            raise LangDependencyError("Error Correction - language not implemented for error correction")
        elif self.lang == "english":
            raise LangDependencyError("Error Correction - language not implemented for error correction")
        elif self.lang == "italian":
            raise LangDependencyError("Error Correction - language not implemented for error correction")

        return text

    def spanish_negation(self, text):
        """
        Standarizes negation sentences, nouns are also considering with the operator "sin" (without)
        Markers like ninguno, ningún, nadie are considered as another word.
        """
        if getattr(self, 'skip_words', None) is None:
            self.skip_words = "me|te|se|lo|les|le|los"
            self.skip_words = self.skip_words + "|" + "|".join(self.neg_stopwords)

        text = text.replace('~', ' ')
        tags = _sURL_TAG + "|" + _sUSER_TAG + "|" + _sENTITY_TAG + "|" + \
               _sHASH_TAG + "|" + \
               _sNUM_TAG + "|" + _sNEGATIVE + "|" + \
               _sPOSITIVE + "|" + _sNEUTRAL + "|"
        
        # unifies negation markers under the "no" marker 
        text = re.sub(r"\b(jam[aá]s|nunca|sin|ni|nada)\b", " no ", text, flags=re.I)
        # reduces to unique negation marker        
        text = re.sub(r"\b(jam[aá]s|nunca|sin|no|nada)(\s+\1)+", r"\1", text, flags=re.I)
        p1 = re.compile(r"(?P<neg>((\s+|\b|^)no))(?P<sk_words>(\s+(" +
                        self.skip_words + "|" + tags + r"))*)\s+(?P<text>(?!(" +
                        tags + ")(\s+|\b|$)))", flags=re.I)
        m = p1.search(text)
        if m:
            text = p1.sub(r"\g<sk_words> \g<neg>_\g<text>", text)
        # removes isolated marks "no_" if marks appear because of negation rules
        text = re.sub(r"\b(no_)\b", r" no ", text, flags=re.I)
        # removes extra spaces because of transformations 
        text = re.sub(r"\s+", r" ", text, flags=re.I)
        return text.replace(' ', '~')

    def english_negation(self, text):
        """
        Standarizes negation sentences
        markers used:
                     "not, no, never, nor, neither"
                     "any" is only used with negative sentences.  
        """
        
        if getattr(self, 'skip_words', None) is None:
            self.skip_words = "me|you|he|she|it|us|the"
            self.skip_words = self.skip_words + "|" + "|".join(self.neg_stopwords)
            
        text = text.replace('~', ' ')
        tags = _sURL_TAG + "|" + _sUSER_TAG + "|" + _sENTITY_TAG + "|" + \
            _sHASH_TAG + "|" + \
            _sNUM_TAG + "|" + _sNEGATIVE + "|" + \
            _sPOSITIVE + "|" + _sNEUTRAL + "|"
  
        # expands contractions of negation
        text = re.sub(r"\b(ca)n't\b", r"\1n not ", text, flags=re.I)
        text = re.sub(r"\b(w)on't\b", r"\1ill not ", text, flags=re.I)
        text = re.sub(r"\b(sha)n't\b", r"\1ll not ", text, flags=re.I)
        text = re.sub(r"\b(can)not\b", r"\1 not ", text, flags=re.I)
        text = re.sub(r"\b([a-z]+)(n't)\b", r"\1 not ", text, flags=re.I)

        # checks negative sentences with the "any" marker and changes "any" to "not" makers
        pp1 = re.compile(r"(?P<neg>(\bnot\b))(?P<text>(\s+([^\s]+?)\s+)+?)(?P<any>any\b)", flags=re.I)
        m = pp1.search(text)
        if m:
            text = pp1.sub(r"\g<neg> \g<text> not ", text)
            
        # unifies negation markers under the "not" marker
        # markers used:
        #              not, no, never, nor, neither
        text = re.sub(r"\b(not|no|never|nor|neither)\b", r" not ", text, flags=re.I)
        text = re.sub(r"\s+", r" ", text, flags=re.I)

        p1 = re.compile(r"(?P<neg>((\s+|\b|^)not))(?P<sk_words>(\s+(" + \
                        self.skip_words + "|" + tags + r"))*)\s+(?P<text>(?!(" + \
                        tags + ")(\s+|\b|$)))", flags=re.I)
        m = p1.search(text)
        if m:
            text = p1.sub(r"\g<sk_words> \g<neg>_\g<text>", text)
        # removes isolated marks "no_" if marks appear because of negation rules
        text = re.sub(r"\b(not_)\b", r" not ", text, flags=re.I)
        text = re.sub(r"\s+", r" ", text, flags=re.I)
        return text.replace(' ', '~')

    def italian_negation(self, text):
        
        
        if getattr(self, 'skip_words', None) is None:
            self.skip_words = "mi|ti|lo|gli|le|ne|li|glieli|glielo|gliela|gliene|gliele"
            self.skip_words = self.skip_words + "|" + "|".join(self.neg_stopwords)
            
        
       
        text = text.replace('~', ' ')
        tags = _sURL_TAG + "|" + _sUSER_TAG + "|" + _sENTITY_TAG + "|" + \
               _sHASH_TAG + "|" + \
               _sNUM_TAG + "|" + _sNEGATIVE + "|" + \
               _sPOSITIVE + "|" + _sNEUTRAL + "|"
        
 
        # unifies negation markers under the "no" marker                
        text = re.sub(r"\b(mai|senza|non|no|né|ne)\b", " no ", text, flags=re.I)
        
        # reduces to unique negation marker         
        text = re.sub(r"\b(mai|senza|non|no|né|ne)(\s+\1)+", r"\1", text, flags=re.I)
        
        p1 = re.compile(r"(?P<neg>((\s+|\b|^)no))(?P<sk_words>(\s+(" +
                        self.skip_words + "|" + tags + r"))*)\s+(?P<text>(?!(" +
                        tags + ")(\s+|\b|$)))", flags=re.I)
        
        m = p1.search(text) 
        
        if m:
            text = p1.sub(r"\g<sk_words> \g<neg>_\g<text>", text)
        # removes isolated marks "no_" if marks appear because of negation rules
        text = re.sub(r"\b(no_)\b", r" no ", text, flags=re.I)
        # removes extra spaces because of transformations 
        text = re.sub(r"\s+", r" ", text, flags=re.I)
        return text.replace(' ', '~')

    # DOUGLAS - portuguese_negation()
    def portuguese_negation(self, text):
        """
        Standarizes negation sentences, nouns are also considering with the operator "sin" (without)
        Markers like ninguno, ningún, nadie are considered as another word.
        """
        if getattr(self, 'skip_words', None) is None:
            self.skip_words = "eu|voce|isso|isto|o que|ele|ela|eles|elas|nos|o|a"
            # DOUGLAS - TODO Implement neg.stopwords list in the files
            self.skip_words = self.skip_words + "|" + "|".join(self.neg_stopwords)

        text = text.replace('~', ' ')
        tags = _sURL_TAG + "|" + _sUSER_TAG + "|" + _sENTITY_TAG + "|" + \
               _sHASH_TAG + "|" + \
               _sNUM_TAG + "|" + _sNEGATIVE + "|" + \
               _sPOSITIVE + "|" + _sNEUTRAL + "|"

        print "Text in negation:"
        print text
        
        # unifies negation markers under the "nao" marker 
        text = re.sub(r"\b(jam[aá]is|nunca|sem|nem|nada)\b", " nao ", text, flags=re.I)
        # reduces to unique negation marker        
        text = re.sub(r"\b(jam[aá]is|nunca|sem|nao|nada)(\s+\1)+", r"\1", text, flags=re.I)
        p1 = re.compile(r"(?P<neg>((\s+|\b|^)nao))(?P<sk_words>(\s+(" +
                        self.skip_words + "|" + tags + r"))*)\s+(?P<text>(?!(" +
                        tags + ")(\s+|\b|$)))", flags=re.I)
        m = p1.search(text)
        if m:
            text = p1.sub(r"\g<sk_words> \g<neg>_\g<text>", text)
        # removes isolated marks "nao_" if marks appear because of negation rules
        text = re.sub(r"\b(nao_)\b", r" nao ", text, flags=re.I)
        # removes extra spaces because of transformations 
        text = re.sub(r"\s+", r" ", text, flags=re.I)
        return text.replace(' ', '~')

    def filterStopWords(self, text, stopwords_option):
        if stopwords_option != 'none':
            for sw in self.stopwords:
                if stopwords_option == 'delete':
                    text = re.sub(r"\b(" + sw + r")\b", r"~", text, flags=re.I)
                elif stopwords_option == 'group':
                    text = re.sub(r"\b(" + sw + r")\b", r"~_sw~", text, flags=re.I)

        return text

    # DOUGLAS - Filter stop words also by PoS tag
    def filter_stopwords_pos(self, text, stopwords_option):
        stopwords_option = 'group'
        tokens = re.split(r"~", text.strip()) # Text has the char "~" to indicate the space between tokens
        t = []
        if stopwords_option == 'delete':
            for tok in tokens:
                token, tag = tok.split("/")
                if token in self.stopwords or tag.startswith("D") or tag.startswith("P"):
                    continue
                else:
                    t.append(tok)
        elif stopwords_option == 'group':
            for tok in tokens:
                token, tag = tok.split("/")
                if token in self.stopwords or tag.startswith("D") or tag.startswith("P"):
                    t.append("_sw" + "/" + tag)
                else:
                    t.append(tok)

        return "~".join(t)

    def filter_entities(self, text):
        tokens = re.split(r"~", text.strip()) # Text has the char "~" to indicate the space between tokens
        t = []
        for tok in tokens:
            token, tag = tok.split("/")
            if tag != "NP0000":
                t.append(tok)

        return "~".join(t)

    def portuguese_correction(self, text):
        # DOUGLAS - Check if each word is valid from the Portuguese dictionary from Freeling
        tokens = re.split(r"~", text.strip()) # Text has the char "~" to indicate the space between tokens
        t = []
        for tok in tokens:
            if tok in self.dictionary_words:
                t.append(tok)
                continue
            elif tok.strip() in self.abbreviation_words.keys():
                expansion = self.abbreviation_words[tok]
                tokens_in_expansion = expansion.split(" ")
                for token in tokens_in_expansion:
                    t.append(token)

                continue
            elif len(re.findall("a{2,}|b{3,}|c{3,}|d{3,}|e{2,}|f{3,}|g{3,}|h{3,}|i{2,}|j{3,}|k{3,}|l{3,}|m{3,}|n{3,}|o{2,}|p{3,}|q{3,}|r{3,}|s{3,}|t{3,}|u{2,}|v{3,}|x{3,}|w{3,}|y{3,}|z{3,}", tok)) > 0:
                lengthenings = lengthenings = re.findall("a{2,}|b{3,}|c{3,}|d{3,}|e{2,}|f{3,}|g{3,}|h{3,}|i{2,}|j{3,}|k{3,}|l{3,}|m{3,}|n{3,}|o{2,}|p{3,}|q{3,}|r{3,}|s{3,}|t{3,}|u{2,}|v{3,}|x{3,}|w{3,}|y{3,}|z{3,}", tok)
                # DOUGLAS - TODO Add exceptions such as "carro", or consider the size of the lengthening. Maybe implement reduction of letters and checking
                # in the dict per letter. Add the option of intensification by allowing two repeated letters.
                for lengthening in lengthenings:
                    tok = re.subn(lengthening, lengthening[0], tok)[0]

                if tok in self.dictionary_words:
                    t.append(tok)
                    continue

            
            print "ERROR TOKEN: %s" % tok

            if re.search("nb", tok, re.IGNORECASE):
                tok = re.sub("nb", "mb", tok, re.IGNORECASE)
            if re.search("np", tok, re.IGNORECASE):
                tok = re.sub("np", "mp", tok, re.IGNORECASE)
            if re.search("ss[a|e|i|o|u]c", tok, re.IGNORECASE):
                vowel = re.findall("ss([a|e|i|o|u])c", tok, re.IGNORECASE)[0]
                tok = re.sub("ss[a|e|i|o|u]c", "c" + vowel + "ss", tok, re.IGNORECASE)
            if re.search("^lej", tok, re.IGNORECASE):
                tok = re.sub("^lej", "leg", tok, re.IGNORECASE)
            if re.search("^rej", tok, re.IGNORECASE) and re.search("^rejei", tok, re.IGNORECASE) == None:
                tok = re.sub("^rej", "reg", tok, re.IGNORECASE)
            if re.search("^alj", tok, re.IGNORECASE):
                tok = re.sub("^alj", "alg", tok, re.IGNORECASE)
            if re.search("[a-z]+sinho$", tok, re.IGNORECASE):
                tok = re.sub("sinho$", "zinho", tok, re.IGNORECASE)
            if re.search("[a-z]+sinha$", tok, re.IGNORECASE):
                tok = re.sub("sinha$", "zinha", tok, re.IGNORECASE)
            if re.search("[a-z]+sito$", tok, re.IGNORECASE):
                tok = re.sub("sito$", "zito", tok, re.IGNORECASE)
            if re.search("[a-z]+sita$", tok, re.IGNORECASE):
                tok = re.sub("sita$", "zita", tok, re.IGNORECASE)
            # DOUGLAS - TODO: Add exceptions
            if re.search("[b|c|d|f|g|h|j|k|l|m|n|p|q|r|s|t|v|w|x|y|z]{1}[a|e|o]{1}[i|u]ch", tok, re.IGNORECASE):
                tok = re.sub("ch", "x", tok, re.IGNORECASE)
            if re.search("[a-z]+anx", tok, re.IGNORECASE):
                tok = re.sub("anx", "anch", tok, re.IGNORECASE)
            if re.search("[a-z]+inx", tok, re.IGNORECASE):
                tok = re.sub("inx", "inch", tok, re.IGNORECASE)
            if re.search("[a-z]+onx", tok, re.IGNORECASE):
                tok = re.sub("onx", "onch", tok, re.IGNORECASE)
            if re.search("[a-z]+unx", tok, re.IGNORECASE):
                tok = re.sub("unx", "unch", tok, re.IGNORECASE)
            

            print "ERROR TOKEN CORRECTED: %s" % tok
            t.append(tok)

        return "~".join(t)

    # DOUGLAS - Performs lemmatizing plus Part-of-Speech tagging, provinding word/pos_tag
    def portuguese_lemmatizing(self, text):
        tokens = re.split(r"~", text.strip())
        new_text = " ".join(tokens)
        t = []

        analyzer = Analyzer(config='pt.cfg', lang='pt')
        xml = analyzer.run(new_text, 'noflush')
        xml_string = etree.tostring(xml)
        #print "XML_STRING"
        #print xml_string[]
        print xml_string

        y = BeautifulSoup(xml_string, "lxml")

        #print "Y BEAUTISOUP"
        #print y



        total_tokens = len(y.sentences.sentence.findAll("token"))

        #print total_tokens

        for i in range(0,total_tokens):
            #lemma = y.sentences.sentence.findAll("token")[i].analysis["lemma"]
            #pos = y.sentences.sentence.findAll("token")[i].analysis["tag"]
            lemma = y.sentences.sentence.findAll("token")[i]["lemma"]
            if re.search(lemma, "/"):
                lemma = y.sentences.sentence.findAll("token")[i]["form"]

            pos = y.sentences.sentence.findAll("token")[i]["tag"]
            new_token = "/".join([lemma, pos])
            t.append(new_token)

        #sys.exit()

        return "~".join(t)

    # DOUGLAS - Remove lexical information
    def remove_lexical_info(self, text):
        tokens = re.split(r"~", text.strip())
        t = []
        for tok in tokens:
            token, tag = tok.split("/")
            t.append(token)

        return "~".join(t)
    
    def transform(self, text, negation=False, stemming=False, stopwords=OPTION_NONE):

        if self.correction:
            text = self.error_correction(text)

        print "----------------------------------------------------------------------------------------"
        print "Text after Error Correction:"
        print text

        if stemming:
            text = self.stemming(text)

        print "----------------------------------------------------------------------------------------"
        print "Text after Stemming:"
        print text

        if self.lem:
            text = self.lemmatizing(text)

        print "----------------------------------------------------------------------------------------"
        print "Text after Lemmmatizing:"
        print text

        if negation:
            text = self.negation(text)

        print "----------------------------------------------------------------------------------------"
        print "Text after Negation:"
        print text

        if self.del_ent:
            text = self.filter_entities(text)

        print "----------------------------------------------------------------------------------------"
        print "Text after Entitities Filtering:"
        print text

        text = self.filter_stopwords_pos(text, stopwords)

        print "----------------------------------------------------------------------------------------"
        print "Text after Stopwords filtering:"
        print text

        text = self.remove_lexical_info(text)

        print "----------------------------------------------------------------------------------------"
        print "Text after Lexical Info Removal:"
        print text

        #sys.exit()

        return text
