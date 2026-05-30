from enum import Enum
from pathlib import Path
from typing import Tuple, List
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, TensorDataset, DataLoader, random_split

VAL_SIZE = 0.2

class Participant(Enum):
    P1 = 'P1'
    P2 = 'P2'
    P3 = 'P3'
    P4 = 'P4'
    P5 = 'P5'
    P6 = 'P6'
    P7 = 'P7'
    P8 = 'P8'
    P9 = 'P9'
    P10 = 'P10'
    P11 = 'P11'
    P12 = 'P12'
    P13 = 'P13'
    P14 = 'P14'
    P15 = 'P15'
    P16 = 'P16'
    P17 = 'P17'
    P18 = 'P18'
    P19 = 'P19'

    def to_num(self) -> str:
        return self.value.replace('P', '')



class DayT1T2(Enum):
    T1 = 'T1'
    T2 = 'T2'

    def to_num(self) -> str:
        return self.value.replace('T', '')

    def other_day(self):
        return DayT1T2.T1 if self == DayT1T2.T2 else DayT1T2.T2

def load_X_y(root: Path, participant: Participant, experiment_day: DayT1T2) -> Tuple[np.ndarray, np.ndarray]:
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    arrays = []
    for filename in root.glob(f'*{participant.value}_{experiment_day.value}_X.npy'):
        with open(filename.as_posix(), 'rb') as f:
            arr = np.load(f)
            arrays.append(arr)
    X = np.concatenate(arrays, axis=0)

    labels = []
    for filename in root.glob(f'*{participant.value}_{experiment_day.value}_y.npy'):
        with open(filename.as_posix(), 'rb') as f:
            l = np.load(f)
            labels.append(l)
    y = np.concatenate(labels, axis=0)

    assert X.shape[0] == y.shape[0], f'{X.shape[0]} != {y.shape[0]}'
    return X, y

def get_same_split_day_arrays(root: Path,
                              participant: Participant,
                              day: DayT1T2,
                              split_ratio: float = 0.8,
                              shuffle: bool = True, scale: bool = True):
    assert root.exists(), f'{root} does not exist'
    assert root.is_dir(), f'{root} is not a directory'

    X, y = load_X_y(root, participant, day)

    num_recordings = len(X)
    assert num_recordings == len(y), f'Expected {num_recordings} labels, but got {len(y)}'

    if shuffle:
        shuffled_indices = torch.randperm(num_recordings)
        X, y = X[shuffled_indices], y[shuffled_indices]

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    split_index = int(split_ratio * num_recordings)

    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    return X_train, y_train, X_test, y_test

def get_same_split_day_datasets(root: Path,
                                participant: Participant,
                                day: DayT1T2,
                                split_ratio: float = 0.8,
                                shuffle: bool = True,
                                scale: bool = True,
                                features_inds: List[int] = None,
                                channels_inds: List[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates split datasets for train and test by processing and optionally reshaping input data.

    This function loads data for a specific participant and day, applies a train-test split,
    and optionally reshapes or filters the features or channels based on provided indices.
    The processed data is returned in the form of PyTorch tensors for training and testing.

    Args:
        root (Path): The root directory containing the dataset files.
        participant (Participant): The participant whose data is to be processed.
        day (DayT1T2): The specific day of data collection for the participant.
        split_ratio (float): The proportion of the data to be used for training. Default is 0.8.
        shuffle (bool): Whether to shuffle the data before splitting. Default is True.
        scale (bool): Whether to scale the data during processing. Default is True.
        features_inds (List[int]): Optional list of feature indices to filter during processing.
        channels_inds (List[int]): Optional list of channel indices to filter during processing.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing four tensors:
            - X_train (torch.Tensor): Training feature data.
            - y_train (torch.Tensor): Training target labels.
            - X_test (torch.Tensor): Testing feature data.
            - y_test (torch.Tensor): Testing target labels.
    """

    X_train, y_train, X_test, y_test = get_same_split_day_arrays(root, participant, day, split_ratio, shuffle, scale)
    if features_inds is not None:
        X_train = X_train.reshape(-1, 16, 20)[..., features_inds].reshape(-1, 16 * len(features_inds))
        X_test = X_test.reshape(-1, 16, 20)[..., features_inds].reshape(-1, 16 * len(features_inds))
    if channels_inds is not None:
        X_train = X_train.reshape(-1, 16, 16)[:, channels_inds, :].reshape(-1, 16 * len(channels_inds))
        X_test = X_test.reshape(-1, 16, 16)[:, channels_inds, :].reshape(-1, 16 * len(channels_inds))


    return torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long(), torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()

    # return (TensorDataset(torch.from_numpy(X_train).float(),
    #                       torch.from_numpy(y_train).long()),
    #         TensorDataset(torch.from_numpy(X_test).float(),
    #                       torch.from_numpy(y_test).long()))


def get_dataloaders(args):
    train_size, val_size = 1-VAL_SIZE, VAL_SIZE
    keep_features, keep_channels = None, None
    if args.num_features_per_channel == 16:
        remove_features = [3,4,5,9]
        keep_features = [i for i in range(20) if i not in remove_features]
    if args.num_features // args.num_features_per_channel == 8:
        keep_channels = [0,1,2,3,4,5,6,7]

    id = 0
    x_train_list, y_train_list, x_test_list, y_test_list = [], [], [], []
    for p in Participant:
        for d in DayT1T2:
            # train_dataset, test_dataset = get_split_between_days_dataset(root=Path(args.data_path), participant=p,
            #                                                              train_day=DayT1T2.T1, scale=True,
            #                                                              features_inds=keep_features,
            #                                                              channels_inds=keep_channels)

            x_train, y_train, x_test, y_test = get_same_split_day_datasets(root=Path(args.data_root), participant=p,
                                                                      day=d, scale=True, split_ratio=0.6,
                                                                      features_inds=keep_features,
                                                                      channels_inds=keep_channels)
            x_train_list.append(x_train)
            y_train_list.append(y_train)
            x_test_list.append(x_test)
            y_test_list.append(y_test)

    dataset = TensorDataset(torch.cat(x_train_list, dim=0), torch.cat(y_train_list, dim=0))
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    test_dataset = TensorDataset(torch.cat(x_test_list, dim=0), torch.cat(y_test_list, dim=0))

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=2)


    return train_loader, val_loader, test_loader