#!/usr/bin/env python
# coding: utf-8
# The script is based on and has some minor changes to the original version written by Hengameh.

import csv
import preprocessor as p
from preprocessor import api
import pandas as pd
import numpy as np
from twokenize import *
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.dicts.emoticons import emoticons
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.classes.spellcorrect import SpellCorrector
import re
from langdetect import detect
from tqdm import tqdm
import nltk
from cleantext import clean
import spacy
from spacy.lang.en import English

social_tokenizer = SocialTokenizer(lowercase=False).tokenize
spell_corrector = SpellCorrector(corpus="english")


def extract_url(row, min_len_url=10):
    if len(row['rt_urls_list']) > min_len_url:
        tweet_url = row['rt_urls_list'].split(',')[1].split('\'')[-2]
    else:
        tweet_url = 'None'
    return tweet_url


class SentClean:
    prep_default = {'spell': False,
                    'remove_sequences': False,
                    'lowercase': False,
                    'punctuations': [],
                    'excluding_criteria': ['copyright','copyright','medrxiv','appendix'],
                    'starting_keywords_to_remove': [
                        'method', 'results', 'result', 'conclusion', 'conclusions', 'evaluation', 'evaluations',
                        'objectives', 'objective', 'cc - by international license', 'doi']
                    }
    def __init__(self,
                 prep=prep_default
                 ):
        """
        Constructor of clean functions over extracted texts/tweets
        :param prep: paramter settings of the text-preprocessor
        """

        # check existence of the keys within prep dict, which needs to be a list
        for k in self.prep_default.keys():
            if not k in prep.keys():
                prep[k] = self.prep_default[k]

        self.prep = prep
        self.omit = list(emoticons.keys()) + list(emoticons.values())
        self.text_processor = TextPreProcessor(
            fix_html=True,
            normalize=[],
            segmenter='twitter',
            corrector='twitter',
            fix_text=True,
            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=prep['spell'],  # Unpack contractions (can't -> can not)
            spell_correction=prep['spell'],
            spell_correct_elong=prep['spell'],
            tokenizer=SocialTokenizer(lowercase=prep['lowercase']).tokenize,
            dicts=[{}],
            omit=list(emoticons.keys()) + list(emoticons.values()),
        )
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp_sent = English()  # just the language with no model
        sentencizer = self.nlp_sent.create_pipe("sentencizer")
        self.nlp_sent.add_pipe(sentencizer)



    def clean_tweet(self, text):
        text_list = self.clean_sentences([text])
        if not len(text_list) > 0:
            return None

        return text_list


    def pattern_repl(self, matchobj):
        """
        Return a replacement string to be used for match object
        """
        return ' '.rjust(len(matchobj.group(0)))


    def clean_sentences(self, sentences, min_len_sent=5, max_num_punctuations=10):
        """
        function to clean list of sentences
        :param sentences: list (str) of the sentences
        :param min_len_sent: int parameter used to trim
        :param max_num_punctuations: int parameter used to trim
        :return: cleaned sentences
        """

        # remove non english sentences
        en_sentences = []
        for s in sentences:
            try:
                if detect(s) == 'en':
                    en_sentences += [s]
            except:
                continue
        sentences = en_sentences

        # remove chinese characters
        sentences = [re.sub("([^\x00-\x7F])+", " ", text) for text in sentences]

        # restrict to ascii characters
        sentences = [s.encode('ascii', errors='ignore').decode() for s in sentences]
        # print(f'input sentence is {sentences}.')
        # trim length
        trim_sentences = [s for s in sentences if len(s.split()) > min_len_sent]
        new_trim_sentences = []
        for s in trim_sentences:
            p_count = 0
            for p_ in self.prep['punctuations']:
                p_count += s.count(p_)
            if p_count < max_num_punctuations:
                new_trim_sentences += [s]

        # remove redundant
        trim_sentences = list(set(new_trim_sentences))
        # print(f"trim sentence is: {trim_sentences}.")
        # extra sentence wise pre processing steps
        new_text_list = []
        for sent_ in trim_sentences:
            # space correction on urls
            text = sent_.replace('http: /', 'https:/')
            text = text.replace('https: /', 'https:/')
            text = p.clean(text)
            text = clean(text,
                         fix_unicode=True,  # fix various unicode errors
                         to_ascii=True,  # transliterate to closest ASCII representation
                         lower=True,  # lowercase text
                         no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
                         no_urls=True,  # replace all URLs with a special token
                         no_emails=True,  # replace all email addresses with a special token
                         no_phone_numbers=True,  # replace all phone numbers with a special token
                         no_numbers=False,  # replace all numbers with a special token
                         no_digits=False,  # replace all digits with a special token
                         no_currency_symbols=False,  # replace all currency symbols with a special token
                         no_punct=False,  # fully remove punctuation
                         replace_with_url=" ",
                         replace_with_email=" ",
                         replace_with_phone_number=" ",
                         replace_with_number=" ",
                         replace_with_digit=" ",
                         replace_with_currency_symbol=" ",
                         lang="en"  # set to 'de' for German special handling
                         )
            # remove citations
            text = re.sub('\[(\s*\d*,*\s*)+\]', '', text)
            text = re.sub('\[(\s*\d*-*\s*)+\]', '', text)
            text = re.sub('\((\s*\d*,*\s*)+\)', '', text)
            text = re.sub('\((\s*\d*-*\s*)+\)', '', text)

            # replace [**Patterns**] with spaces.
            text = re.sub(r'\[\*\*.*?\*\*\]', self.pattern_repl, text)
            # remove hashtag symbol and unpack it
            text = " ".join(self.text_processor.pre_process_doc(text))
            # remove emoticons
            for item in self.omit:
                text = text.replace(item, ' ')
            # remove non-word character-repetitions
            text = re.sub(r'(\W)\1+', r'\1', text)
            if self.prep['remove_sequences']:
                # remove sequences like 'A p p e n d i x'
                text = re.sub(r'(\S\s){3,}', '', text)
            for p_ in self.prep['punctuations']:
                # replace `_` with spaces.
                text = text.replace(p_, ' ' + p_ + ' ')
            if self.prep['spell']:
                # spell correction
                text = " ".join(spell_corrector.correct(w) for w in social_tokenizer(text))
            # remove douplicated whitespaces
            text = squeezeWhitespace(text)
            
            if text.split(' ')[0] in self.prep['starting_keywords_to_remove'] and text.split(' ')[1] == ':':
                text = ' '.join(text.split(' ')[2:])
            # exclude sentences including keywords like
            for ex_key in self.prep['excluding_criteria']:
                if ex_key in text:
                    text = ''

            # check if there exists verb on the text
            doc = self.nlp((text))
            number_of_verbs = len([token.lemma_ for token in doc if token.pos_ == "VERB"])
            # print(f"the text is: {text}.")
            if len(text.split(' ')) > min_len_sent and  number_of_verbs > 0:
                new_text_list.append(text)

        return new_text_list


if __name__ == "__main__":
    preprocessing_options = {'spell': False, 'remove_sequences': False, 'punctuations': []}
    clean_sentences = SentClean(prep=preprocessing_options).clean_sentences
    sentence_1 = 'Sentence A: ok Xiao, you can use this function to clear your list of sentences'
    sentence_2 = 'Sentence B: and this is another sentence and this is another sentence'
    new_sent_list = clean_sentences([sentence_1, sentence_2], 
                                    min_len_sent=10, 
                                    max_num_punctuations=10)
    print(new_sent_list)
