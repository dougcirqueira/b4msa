# -*- coding: utf-8 -*-
# Copyright 2016 Mario Graff (https://github.com/mgraffg)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def test_tweet_iterator():
    import os
    import gzip
    from b4msa.utils import tweet_iterator
    
    fname = os.path.dirname(__file__) + '/text.json'
    a = [x for x in tweet_iterator(fname)]
    fname_gz = fname + '.gz'
    with open(fname, 'r') as fpt:
        with gzip.open(fname_gz, 'w') as fpt2:
            fpt2.write(fpt.read().encode('ascii'))
    b = [x for x in tweet_iterator(fname_gz)]
    assert len(a) == len(b)
    for a0, b0 in zip(a, b):
        assert a0['text'] == b0['text']
    os.unlink(fname_gz)


def test_textmodel():
    from b4msa.textmodel import TextModel
    from b4msa.utils import tweet_iterator
    import os
    fname = os.path.dirname(__file__) + '/text.json'
    tw = list(tweet_iterator(fname))
    text = TextModel([x['text'] for x in tw])
    # print(text.tokenize("hola amiguitos gracias por venir :) http://hello.com @chanfle"))
    # assert False
    assert isinstance(text[tw[0]['text']], list)


def test_params():
    import os
    import itertools
    from b4msa.params import BASIC_OPTIONS
    from b4msa.textmodel import TextModel
    from b4msa.utils import tweet_iterator

    params = dict(strip_diac=[True, False], usr_option=BASIC_OPTIONS,
                  url_option=BASIC_OPTIONS)
    params = sorted(params.items())
    fname = os.path.dirname(__file__) + '/text.json'
    tw = [x for x in tweet_iterator(fname)]
    text = [x['text'] for x in tw]
    for x in itertools.product(*[x[1] for x in params]):
        args = dict(zip([x[0] for x in params], x))
        ins = TextModel(text, **args)
        assert isinstance(ins[text[0]], list)

def test_emoticons():
    from b4msa.textmodel import EmoticonClassifier, norm_chars
    emo = EmoticonClassifier()
    for a, b in [
            ("Hi :) :P XD", "~Hi~_pos~_pos~_pos~"),
            ("excelente dia xc", "~excelente~dia~_neg~")
    ]:
        _a = norm_chars(a)
        assert ' ' not in _a, "norm_chars normalizes spaces {0} ==> {1}".format(a, _a)
        _b = emo.replace(_a)
        print("[{0}] => [{1}]; should be [{2}]".format(a, _b, b))
        assert _b == b


def test_lang():
    from b4msa.textmodel import TextModel

    #text = [
    #    "Hi :) :P XD",
    #    "excelente dia xc",
    #    "el alma de la fiesta XD"
    #]
    text = ["vish, nada desse carro! tomar Eguaa! vai toooomar no cuuuuu pq jamais vou deixar d lado! Realmente sem pe nem cabbbecaaa!! :("]

    model = TextModel(text, **{
        "del_dup1": True,
        "emo_option": "group",
        "lc": True,
        "negation": True,
        "num_option": "group",
        "stemming": True,
        "stopwords": "group",
        "strip_diac": True,
        "token_list": [
            -1,
            # 5,
        ],
        "url_option": "group",
        "usr_option": "group",
        "lang": "portuguese",
    })
    #text = "El alma de la fiesta :) conociendo la maquinaria @user bebiendo nunca manches que onda"
    text = "vish, nada desse carro! tomar Eguaa! vai toooomar no cuuuuu pq jamais vou deixar d lado! Realmente sem pe nem cabbbecaaa!! :("
    a = model.tokenize(text)
    b = ['_sw', 'alma', '_sw', '_sw', 'fiest', '_pos', 'conoc', '_sw', 'maquinari', '_usr', 'beb', 'no_manch', '_sw', 'onda']
    print a
    #print(text)
    #assert a == b, "got: {0}, expected: {1}".format(a, b)


def test_negations():
    from b4msa.textmodel import TextModel

    text = [
        "el alma de la fiesta XD"
    ]
    model = TextModel(text, **{
        'num_option': 'group',
        'strip_diac': False,
        'stopwords': 'delete',
        'negation': True,
        'stemming': True,
        'lc': False, 'token_list': [-1],
        'usr_option': 'group', 'del_dup1': False, 'emo_option': 'group', 'lang': 'spanish', 'url_option': 'delete'
    })

    text = """@usuario los pollos y las vacas nunca hubiesen permitido que no se hubiese hecho nada al respecto"""
    a = model.tokenize(text)
    b = ['_usr', 'pol', 'vac', 'hub', 'no_permit', 'hub', 'no_hech', 'no_respect']
    assert a == b


def test_negations_italian():
    from b4msa.textmodel import TextModel

    text = [
        "XD"
    ]

    model = TextModel(text, **{
        'num_option': 'group',
        'strip_diac': False,
        'stopwords': 'delete',
        'negation': True,
        'stemming': True,
        'lc': False, 'token_list': [-1],
        'usr_option': 'group',
        'del_dup1': False,
        'emo_option': 'group',
        'lang': 'italian',
        'url_option': 'delete'
    })

    text = """@User Come non condividere; me ne frega niente"""
    a = model.tokenize(text)
    print("Input:", text)
    print("Output:", a)
    b = ['_usr', 'com', 'no_condividere', 'me', 'no_freg', 'nient']
    assert a == b

if __name__ == '__main__':
    test_lang()