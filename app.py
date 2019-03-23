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


def load_file(path):
    file_map = {
        ".txt": load_txt,
        ".pdf": load_pdf
    }
    loader = file_map[os.path.splitext(path)[1]]
    return loader(path)


def fresh_model():
    ws = os.cpu_count()
    return Doc2Vec([load_txt(data_dir("lorem ipsum.txt"))], vector_size=5, window=2, min_count=1, workers=ws)


def train_with_dir(model, path, epochs=1):
    docs = os.listdir(path)
    model.train((load_file(os.path.join(path, d)) for d in docs), total_examples=len(docs), epochs=epochs)
    return model
