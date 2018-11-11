from src.dico_builder import build_dictionary
import numpy as np
import io
import argparse
import torch
import pickle
from src.utils import bool_flag, initialize_exp

parser = argparse.ArgumentParser(description='Unsupervised training')
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--params", type=str, default="", help="Reload params")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=10000, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")

# parse parameters
params = parser.parse_args()

def load_vec(emb_path, nmax=50000):
    vectors = []
    word2id = {}
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ')
            assert word not in word2id, 'word found twice'
            vectors.append(vect)
            word2id[word] = len(word2id)
            if len(word2id) == nmax:
                break
    id2word = {v: k for k, v in word2id.items()}
    embeddings = np.vstack(vectors)
    return embeddings, id2word, word2id

if __name__ == '__main__':
    src_word_embs, src_id2word, src_word2id = load_vec(params.src_emb)
    tgt_word_embs, tgt_id2word, tgt_word2id = load_vec(params.tgt_emb)
    src_word_embs = torch.FloatTensor(src_word_embs).cuda()
    tgt_word_embs = torch.FloatTensor(tgt_word_embs).cuda()
    build_dictionary(src_emb=src_word_embs, tgt_emb=tgt_word_embs, params=params)


