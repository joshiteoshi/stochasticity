from stochasticity import Stochasticity, SegmentLabelDataset
from corpus_proc import WordDictionary

import torch
from numpy.random import default_rng

import random

IAMBIC_TRI = "010101"
IAMBIC_TETRA = "01010101"
IAMBIC_PENTA = "0101010101"

COMMON_METER = [IAMBIC_TETRA, IAMBIC_TRI, IAMBIC_TETRA, IAMBIC_TRI]
IAMBIC_PENTAMETER = [IAMBIC_PENTA] * 4 

class PoetryGenerator:

    def __init__(self, model_path, data_path, dict_path):

        torch.backends.cudnn.deterministic = True

        self.model = torch.load(model_path, weights_only=False)
        self.model.eval()

        self.dictionary = WordDictionary(dict_path)
        self.vocab = SegmentLabelDataset(data_path).vocab


    def seq_splitter(self, seq: list[int], success: float=0.8, seed=None) -> list[torch.Tensor]:

        geny = default_rng(seed)

        start_index = 0
        split = []

        while start_index < len(seq):
            if start_index + 1 == len(seq):
                split.append(torch.Tensor(seq[start_index:]).type(torch.int32)) 
                start_index += 1
            
            subseq_len = geny.geometric(success)

            if subseq_len + start_index > len(seq):
                continue

            else:
                split.append(torch.Tensor(seq[start_index:start_index + subseq_len]).type(torch.int32))
                start_index += subseq_len
        
        return split
    
    def label(self, seq: list[list[torch.Tensor]], temperature=1.0, seed=None, retry=7) -> list[str]:

        rng = random.Random(seed)

        stanza = []

        for stress_seq in seq:

            line = []

            for attempt in range(retry):
                label_seq = self.model([stress_seq], mode='decode', temperature=temperature, seed=seed)

                try:
                    for count in range(len(stress_seq)):
                        words = self.dictionary.get_words_from(self.vocab.id2label[label_seq[0][count]], "".join([str(val) for val in stress_seq[count].tolist()]))
                        rng.shuffle(words)
                        line.append(words[0][0])
                    break

                except:
                    line = []
                    if seed is not None:
                        seed += 1
                        rng.seed(seed)
                    if attempt == retry - 1:
                        return ["not satisfiable"]
                    continue

            line = self.det_validator(line)
            stanza.append(' '.join(line))

        return stanza
    
    def set_global_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def generate(self, stresses: list[str], success=0.8, temperature=1.0, seed=None, retry=7):

        if seed is not None:
            self.set_global_seed(seed)
            stress_seqs = [self.seq_splitter([int(c) for c in stress], success, seed + index) for stress, index in zip(stresses, list(range(len(stresses))))]
        else:
            stress_seqs = [self.seq_splitter([int(c) for c in stress], success, seed) for stress in stresses]

        return self.label(stress_seqs, temperature, seed, retry)

    def gen_common_meter(self, success=0.8, temperature=1.0, seed=None):

        return self.generate(COMMON_METER, success, temperature, seed)
    
    def gen_iambic_pentameter(self, success=0.8, temperature=1.0, seed=None):

        return self.generate(IAMBIC_PENTAMETER, success, temperature, seed)
    
    def det_validator(self, line: list[str]):

        for word in range(len(line)):
            if (line[word] == "a" or line[word] == "an") and word != len(line) - 1:
                line[word] = "an" if line[word + 1][0] in "aeiou" else "a"

        return line