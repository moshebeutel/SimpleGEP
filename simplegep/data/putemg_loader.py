import os
import re
from pathlib import Path
from typing import List, Any, Dict
from tqdm import trange
import psutil
import torch
from torch.utils.data import random_split

from simplegep.utils import set_logger


def get_user_list():
    # return ['03', '04', '05']

    return ['03', '04', '05', '06', '07', '08', '09', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
            '22', '23', '24', '25', '26', '27', '29', '30', '31', '33', '34', '35', '36', '38', '39', '42', '43', '45',
            '46', '47', '48', '49', '50', '51', '53', '54']


    # return ['03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
    #         '22', '23', '24', '25', '26', '27', '29', '30', '31', '33', '34', '35', '36', '38', '39', '42', '43', '45',
    #         '46', '47', '48', '49', '50', '51', '53', '54']

def get_features() -> List[Any]:
    # features = ['RMS', 'MAV', 'WL', 'ZC', 'SSC', 'IAV', 'VAR', 'WAMP'] if args.num_features == 8 * 24 else ["IAV",
    #                                                                                                         "AAC",
    #                                                                                                         "DASDV",
    #                                                                                                         "Kurt",
    #                                                                                                         "MAV1",
    #                                                                                                         "MAV2",
    #                                                                                                         "MAV",
    #                                                                                                         "MHW",
    #                                                                                                         'RMS',
    #                                                                                                         "Skew",
    #                                                                                                         "SSI",
    #                                                                                                         'VAR',
    #                                                                                                         'WL',
    #                                                                                                         "MNF",
    #                                                                                                         "MDF",
    #                                                                                                         "PKF",
    #                                                                                                         "MNP",
    #                                                                                                         "TTP",
    #                                                                                                         "VCF",
    #                                                                                                         "OHM"]

    features = ["IAV",
                "AAC",
                "DASDV",
                # "Kurt",
                # "MAV1",
                # "MAV2",
                "MAV",
                "MHW",
                'RMS',
                # "Skew",
                "SSI",
                'VAR',
                'WL',
                "MNF",
                "MDF",
                "PKF",
                "MNP",
                "TTP",
                "VCF",
                "OHM"]
    return features


def get_dataloaders(args):
    import pandas as pd
    from biolab_utilities.putemg_utilities import prepare_data, Record, record_filter, data_per_id_and_date

    logger = set_logger(logger_name=args.sess, log_dir=args.log_root, level=args.log_level)

    # filtered_data_folder = os.path.join(result_folder, 'filtered_data')
    # calculated_features_folder = os.path.join(result_folder, 'calculated_features')
    calculated_features_folder = Path(args.data_root)
    assert calculated_features_folder.exists(), f'{calculated_features_folder} does not exist'
    assert calculated_features_folder.is_dir(), f'{calculated_features_folder} is not a directory'
    assert len(list(
        calculated_features_folder.glob('*.hdf5'))) > 0, f'{calculated_features_folder} does not contain hdf5 files'

    # list all hdf5 files in given input folder
    # all_files = [f.as_posix().replace('_filtered_features', '')
    #              for f in sorted(calculated_features_folder.glob("*_features.hdf5"))]
    all_files = [f.as_posix().replace('_filtered', '')
                 for f in sorted(calculated_features_folder.glob("*_filtered.hdf5"))]

    users_files = []
    users = get_user_list()
    for u in users:
        for f in all_files:
            if f'gestures-{u}' in f:
                users_files.append(f)

    logger.debug(f'{len(users_files)} users files found')

    all_files = users_files

    logger.debug(f'Found {len(all_files)} feature files')

    all_feature_records = [Record(os.path.basename(f)) for f in all_files]

    logger.debug(f'Found {len(all_feature_records)} feature records')

    records_filtered_by_subject = record_filter(all_feature_records)

    logger.debug(f'Filtered {len(records_filtered_by_subject)} records')

    splits_all = data_per_id_and_date(records_filtered_by_subject, n_splits=1)

    logger.debug(f'Splits {len(splits_all)}')

    # load feature data to memory
    dfs: Dict[Record, pd.DataFrame] = {}

    for r in records_filtered_by_subject:
        logger.debug(f'Loading {r.path}')
        logger.debug(f"CPU usage: {psutil.cpu_percent(interval=1)}")
        logger.debug(f"RAM usage: {psutil.virtual_memory().percent}%")
        filename = os.path.splitext(r.path)[0]
        #TODO: remove this
        if filename == 'features_short_time_emg_gestures-10-sequential-2018-04-05-10-14-14-029':
            continue
        dfs[r] = pd.DataFrame(pd.read_hdf(os.path.join(calculated_features_folder, filename + '_filtered.hdf5')))

    features = get_features()

    logger.info(f'Found {len(dfs)} dataframes')

    num_channels = args.num_features // args.num_features_per_channel
    expected_num_channels = [24, 8]
    assert num_channels in expected_num_channels, f'Expected channels one of {expected_num_channels}. Got {num_channels}'
    num_features_per_channel = len(features)
    logger.info(f'Number of channels: {num_channels}')
    logger.info(f'Number of features per channel: {num_features_per_channel}')
    train_size, val_size = 0.8, 0.2

    assert len(
        features) == args.num_features_per_channel, f'Expected {len(features)} features extracted from each channel'
    assert (
                       len(features) * num_channels) == args.num_features, f'Expected num features: {len(features) * num_channels}. Do not match args'

    # defines gestures to be used in shallow learn
    gestures = {
        0: "Idle",
        1: "Fist",
        2: "Flexion",
        3: "Extension",
        4: "Pinch index",
        5: "Pinch middle",
        6: "Pinch ring",
        7: "Pinch small"
    }
    channel_range = {
        "24chn": {"begin": 1, "end": 24},
        # "8chn_1band": {"begin": 1, "end": 8},
        "8chn_2band": {"begin": 9, "end": 16},
        # "8chn_3band": {"begin": 17, "end": 24}
    }
    ch_range = channel_range['24chn' if num_channels == 24 else '8chn_2band' if num_channels == 8 else '8chn_3band']

    num_clients = len(splits_all.values())
    train_x_list, test_x_list = [], []
    train_y_list, test_y_list = [], []

    # for id in range(num_clients // 2):
    for id in trange(num_clients):
        train_x_s, test_x_s = [], []
        train_y_s, test_y_s = [], []

        # for client_id in [2 * id, 2 * id + 1]:
        # iterate over each internal data
        client_id = id
        for i_s, subject_data in enumerate(list(splits_all.values())[client_id]):
            # get data of client
            # prepare training and testing set based on combination of k-fold split, feature set and gesture set
            # this is also where gesture transitions are deleted from training and test set
            # only active part of gesture performance remains
            data = prepare_data(dfs, subject_data, features, list(gestures.keys()))

            logger.debug(f'Processing subject {i_s}:  {subject_data}')
            logger.debug(f'For client: {client_id}')

            # list columns containing only feature data
            regex = re.compile(r'input_[0-9]+_[A-Z]+_[0-9]+')
            cols = list(filter(regex.search, list(data["train"].columns.values)))

            logger.debug(f'Found {len(cols)} columns')

            # strip columns to include only selected channels, eg. only one band
            cols = [c for c in cols if (ch_range["begin"] <= int(c[c.rindex('_') + 1:]) <= ch_range["end"])]
            assert len(cols) == args.num_features, f'Expected cols to contain the features. Got {len(cols)}'

            logger.debug(f'Found {len(cols)} columns after strip')

            # extract limited training x and y, only with chosen channel configuration
            train_x = torch.tensor(data["train"][cols].to_numpy(), dtype=torch.float32)
            train_y = torch.LongTensor(data["train"]["output_0"].to_numpy())
            train_y[train_y > 5] -= 2

            logger.debug(f'Train data shape: {train_x.shape}')

            # # extract limited testing x and y, only with chosen channel configuration
            test_x = torch.tensor(data["test"][cols].to_numpy(), dtype=torch.float32)
            test_y_true = torch.LongTensor(data["test"]["output_0"].to_numpy())
            test_y_true[test_y_true > 5] -= 2

            logger.debug(f'Test data shape: {test_x.shape}')

            # Change order to channel-features instead of feature-channels
            X = train_x.reshape(-1, num_channels, args.num_features_per_channel)
            X = torch.movedim(X, 1, 2)
            train_x = X.reshape(-1, args.num_features)

            X = test_x.reshape(-1, num_channels, args.num_features_per_channel)
            X = torch.movedim(X, 1, 2)
            test_x = X.reshape(-1, args.num_features)

            # mask_train = (train_y < 4)
            # train_x = train_x[mask_train]
            # train_y = train_y[mask_train]
            # mask_train = (train_y != 0)
            # train_x = train_x[mask_train]
            # train_y = train_y[mask_train] - 1

            # mask_test = (test_y_true < 4)
            # test_x = test_x[mask_test]
            # test_y_true = test_y_true[mask_test]
            # mask_test = (test_y_true != 0)
            # test_x = test_x[mask_test]
            # test_y_true = test_y_true[mask_test] - 1

            mask_train = (train_y > 3)
            train_x = train_x[mask_train]
            train_y = train_y[mask_train] - 4

            mask_test = (test_y_true > 3)
            test_x = test_x[mask_test]
            test_y_true = test_y_true[mask_test] - 4

            train_x_s.append(train_x)
            test_x_s.append(test_x)
            train_y_s.append(train_y)
            test_y_s.append(test_y_true)

            logger.debug(f'Train data list length: {len(train_x_s)}')
            logger.debug(f'Test data list length: {len(test_x_s)}')

        train_x_list.append(train_x_s[0])
        test_x_list.append(test_x_s[0])
        train_y_list.append(train_y_s[0])
        test_y_list.append(test_y_s[0])

        # dataset = torch.utils.data.TensorDataset(train_x_s[0], train_y_s[0])
        # train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        #
        # train_loaders[id] = torch.utils.data.DataLoader(
        #     # torch.utils.data.TensorDataset(train_x_s[0], train_y_s[0]),
        #     train_dataset,
        #     shuffle=True,
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers
        # )
        #
        # val_loaders[id] = torch.utils.data.DataLoader(
        #     # torch.utils.data.TensorDataset(train_x_s[1], train_y_s[1]),
        #     val_dataset,
        #     shuffle=False,
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers
        # )
        #
        # test_loaders[id] = torch.utils.data.DataLoader(
        #     torch.utils.data.TensorDataset(test_x, test_y_true),
        #     shuffle=False,
        #     batch_size=args.batch_size,
        #     num_workers=args.num_workers
        # )


    dataset_x = torch.cat(train_x_list, dim=0)
    dataset_y = torch.cat(train_y_list, dim=0)
    dataset = torch.utils.data.TensorDataset(dataset_x, dataset_y)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f'Train data list length: {len(train_dataset)}')
    logger.info(f'Val data list length: {len(val_dataset)}')

    test_x = torch.cat(test_x_list, dim=0)
    test_y_true = torch.cat(test_y_list, dim=0)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y_true)
    logger.info(f'Test data list length: {len(test_dataset)}')

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batchsize, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=args.batchsize, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.batchsize, num_workers=2)

    logger.info(f'Train data loader length: {len(train_loader)}')
    logger.info(f'Val data loader length: {len(val_loader)}')
    logger.info(f'Test data loader length: {len(test_loader)}')

    return train_loader, val_loader, test_loader