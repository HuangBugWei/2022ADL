from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        # instance = {'text': ['switch the language setting over to german', 'how would i go about setting up a direct deposit'], 'intent': ['change_language', 'direct_deposit'], 'id': ['train-345', 'train-1907']}
        # if self.train:
        #     return instance['text'], self.collate_fn(instance)
        # else:
        #     return instance['text'], instance['id']

        return instance
    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # samples: [{'text': 'make a reminder to schedule a tire check', 'intent': 'schedule_maintenance', 'id': 'train-7692'}, {'text': 'can you please talk faster', 'intent': 'change_speed', 'id': 'train-831'}]
        
        x = []
        y = []
        z = []
        for sample in samples:
            x.append(sample['text'])
            y.append(self.label2idx(sample['intent']))
            z.append(sample['id'])
        
        x = self.vocab.encode_batch([sentence.split() for sentence in x])
        
        return x, y, z

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO: implement collate_fn
        raise NotImplementedError
