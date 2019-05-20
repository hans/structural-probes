"""
Builds a structural-probe embedding dataset from spaCy word embeddings.

Refer to the saved hdf5 file with the `ELMO-disk` datatype.
"""

from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import spacy
from tqdm import tqdm

p = ArgumentParser()
p.add_argument("sentences_path", type=Path)
p.add_argument("out_path", type=Path)
p.add_argument("-m", "--spacy_model", type=str, default="en_vectors_web_lg")


def main(args):
    print("Loading spaCy model.")
    nlp = spacy.load(args.spacy_model)

    print("Loading sentences.")
    with args.sentences_path.open("r") as f:
        sentences = [l.strip() for l in f if l.strip()]

    with h5py.File(args.out_path, "w") as out_hf:
        for i, sentence in enumerate(tqdm(sentences)):
            # content is pre-tokenized -- don't allow spaCy to re-tokenize
            doc = nlp.tokenizer.tokens_from_list(sentence.split(" "))

            sentence_vec = np.array([tok.vector for tok in doc]).reshape((1, len(doc), -1))
            out_hf[str(i)] = sentence_vec


if __name__ == "__main__":
    main(p.parse_args())
