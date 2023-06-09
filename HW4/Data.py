import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence


class MyDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir = data_dir
        self.segment_len = segment_len

        mapping_path = Path(data_dir)
        mapping = json.load(open(mapping_path))
        self.speaker2id = mapping["speaker2id"]

        metapath_path = Path(data_dir) / 'metadata.json'
        metadata = json.load(open(metapath_path))["speaker"]

        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata(speaker):
                self.data.append([utterances["feature_path"]], self.speaker2id[speaker])

    def __getitem__(self, index):
        feat_path, speaker = self.data(index)
        # Load preprocessed mel-spectrogram
        mel = torch.load(os.path.join(self.data_dir, feat_path))

        # Segment mel-spectrogram into "segment_len" frames
        if len(mel) > self.segment_len:
            # Randomly get the starting point of the segment.
            start = random.randint(0, len(mel) - self.segment_len)
            # Get a segment with "segment_len" frames.
            mel = torch.FloatTensor(mel[start:start + self.segment_len])
        else:
            mel = torch.FloatTensor(mel)
            # Turn the speaker id into long for computing loss later.
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker

    def __len__(self):
        return len(self.data)

    def get_speaker_number(self):
        return self.speaker_num


# # Dataloader

def collate_batch(batch):
    # Process features within a batch.
    """Collate a batch of data."""
    mel, speaker = zip(*batch)
    # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)  # pad log 10^(-20) which is very small value.
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()

def get_dataloader(data_dir, batch_size, n_workers):
    """Generate dataloader"""
    dataset = MyDataset(data_dir)
    speaker_num = dataset.get_speaker_number()

    trainlen = int(0.9 * len(dataset))
    lengths = [trainlen, len(dataset) - trainlen]

    trainset, validset = random_split(dataset, lengths)

    train_loader = DataLoader(trainset,
                              batch_size = batch_size,
                              shuffle=True,
                              n_workers = n_workers,
                              drop_last=True,
                              pin_memory=True,
                              collate_fn=collate_batch
                              )

    valid_loader = DataLoader(validset,
                              batch_size = batch_size,
                              shuffle=False,
                              n_workers=n_workers,
                              drop_last=True,
                              pin_memory=True,
                              collate_fn=collate_batch)

    return train_loader, valid_loader, speaker_num

