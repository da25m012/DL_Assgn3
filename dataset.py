from collections import Counter
from datasets import load_dataset
import spacy
import torch
from torch.utils.data import Dataset


class Multi30kDataset(Dataset):
    def __init__(self, split='train', src_vocab=None, tgt_vocab=None, max_len=256):
        self.split = split
        self.max_len = max_len

        # load spacy models
        try:
            self.de_nlp = spacy.load("de_core_news_sm")
        except OSError:
            spacy.cli.download("de_core_news_sm")
            self.de_nlp = spacy.load("de_core_news_sm")

        try:
            self.en_nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.en_nlp = spacy.load("en_core_web_sm")

        raw = load_dataset("bentrevett/multi30k")
        self.raw_data = raw[split]

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        self.data = self.process_data()

    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.de_nlp.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.en_nlp.tokenizer(text)]

    def build_vocab(self):
        specials = ["<unk>", "<pad>", "<sos>", "<eos>"]

        de_counter = Counter()
        en_counter = Counter()

        for item in self.raw_data:
            de_counter.update(self.tokenize_de(item["de"]))
            en_counter.update(self.tokenize_en(item["en"]))

        src_vocab = {tok: idx for idx, tok in enumerate(specials)}
        for tok, _ in de_counter.most_common():
            if tok not in src_vocab:
                src_vocab[tok] = len(src_vocab)

        tgt_vocab = {tok: idx for idx, tok in enumerate(specials)}
        for tok, _ in en_counter.most_common():
            if tok not in tgt_vocab:
                tgt_vocab[tok] = len(tgt_vocab)

        return src_vocab, tgt_vocab

    def process_data(self):
        unk_src = self.src_vocab["<unk>"]
        unk_tgt = self.tgt_vocab["<unk>"]
        sos_src = self.src_vocab["<sos>"]
        eos_src = self.src_vocab["<eos>"]
        sos_tgt = self.tgt_vocab["<sos>"]
        eos_tgt = self.tgt_vocab["<eos>"]

        processed = []
        for item in self.raw_data:
            de_tokens = self.tokenize_de(item["de"])
            en_tokens = self.tokenize_en(item["en"])

            src_ids = [sos_src] + [self.src_vocab.get(t, unk_src) for t in de_tokens] + [eos_src]
            tgt_ids = [sos_tgt] + [self.tgt_vocab.get(t, unk_tgt) for t in en_tokens] + [eos_tgt]

            if len(src_ids) <= self.max_len and len(tgt_ids) <= self.max_len:
                processed.append((src_ids, tgt_ids))

        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch, src_pad_idx=1, tgt_pad_idx=1):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [s.size(0) for s in src_batch]
    tgt_lens = [t.size(0) for t in tgt_batch]

    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    padded_src = torch.full((len(batch), max_src), src_pad_idx, dtype=torch.long)
    padded_tgt = torch.full((len(batch), max_tgt), tgt_pad_idx, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_batch, tgt_batch)):
        padded_src[i, :s.size(0)] = s
        padded_tgt[i, :t.size(0)] = t

    return padded_src, padded_tgt
