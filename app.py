import os
from subprocess import check_output

import nltk
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def data_dir(name=None):
    path = os.path.realpath(__file__)
    base = os.path.dirname(path)
    return os.path.join(base, "data", name or "")


def path_to_key(path):
    return os.path.splitext(os.path.basename(path))[0].lower()


def load_txt(path):
    with open(path, 'r') as f:
        return TaggedDocument(
            words=nltk.tokenize.word_tokenize(f.read()),
            tags=[path_to_key(path)]
        )


def load_pdf(path):
    text = check_output(["pdftotext", path, "-"]).decode("utf-8")
    return TaggedDocument(
        words=nltk.tokenize.word_tokenize(text),
        tags=[path_to_key(path)]
    )


EXT_MAP = {
    ".txt": load_txt,
    ".pdf": load_pdf
}


def ext_of(p):
    return os.path.splitext(p)[1]


def load_file(path):
    loader = EXT_MAP[ext_of(path)]
    return loader(path)


def data_from_dir(path):
    docs = os.listdir(path)
    gen = (load_file(os.path.join(path, d)) for d in docs if ext_of(d) in EXT_MAP)
    count = len([d for d in docs if ext_of(d) in EXT_MAP])
    return gen, count


def model_from_dir(path):
    ws = os.cpu_count()
    return Doc2Vec(list(data_from_dir(path)[0]), vector_size=5, window=2, min_count=1, workers=ws)
