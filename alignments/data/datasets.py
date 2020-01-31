from torch.utils.data import Dataset

class ParallelDataset(Dataset):

    def __init__(self, src_file, tgt_file, max_length=-1, min_length=0):
        self.data = []
        with open(src_file) as sf, open(tgt_file) as tf:
            for src, tgt in zip(sf, tf):
                src = src.strip()
                src_length = len(src.split())
                tgt = tgt.strip()
                tgt_length = len(tgt.split())
                if (max_length < 0 or (src_length <= max_length and tgt_length <= max_length)) \
                        and (src_length > min_length and tgt_length > min_length):
                    self.data.append((src, tgt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
