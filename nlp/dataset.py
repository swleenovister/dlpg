import torch
from torch.utils.data import Dataset
from collections import OrderedDict

from common import constant

__all__ = ['WhiteSpaceTokenizer', 'TextClfsDataset']


class TextClfsDataset(Dataset):
    def __init__(self, df, transformers, document_field_name='document', label_field_name='label'):
        self.transformers = transformers
        self.documents = df[document_field_name].tolist()
        self.labels = df[label_field_name].tolist()

        assert len(self.documents) == len(self.labels)

        self.label_to_idx = {l : i for i, l in enumerate(df[label_field_name].unique())}
        self.to_tensor = torch.Tensor


    def __getitem__(self, item_idx):
        assert type(item_idx) == int and item_idx < len(self.labels)
        datapoint = OrderedDict({
            constant.FIELD_NAME_DOCUMENT : self.documents[item_idx],
            constant.FIELD_NAME_LABEL : self.labels[item_idx]
        })

        for transformer in self.transformers:
            print(datapoint)
            datapoint[constant.FIELD_NAME_DOCUMENT] = transformer(datapoint[constant.FIELD_NAME_DOCUMENT])

        datapoint[constant.FIELD_NAME_DOCUMENT] = self.to_tensor(datapoint[constant.FIELD_NAME_DOCUMENT])
        datapoint[constant.FIELD_NAME_LABEL] = self.to_tensor(self.label_to_idx[datapoint[constant.FIELD_NAME_LABEL]])

        return datapoint

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    import pandas as pd
    df = pd.read_csv('../data/nsmc/ratings_test.txt', sep='\t')
    print(df.head(4))

    dataset = TextClfsDataset(df, [WhiteSpaceTokenizer])
    print(dataset[3])