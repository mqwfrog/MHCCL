import os
import random
import re
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Tuple
from sklearn.model_selection import train_test_split
from data_preprocessing.core import split_using_target, split_using_sliding_window
from data_preprocessing.base import BaseDataset, check_path

__all__ = ['WISDM', 'load', 'load_raw']

# Meta Info
SUBJECTS = tuple(range(1, 36+1))
ACTIVITIES = tuple(['Walking', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Downstairs'])
Sampling_Rate = 20 # Hz


class WISDM(BaseDataset):
    def __init__(self, path:Path):
        super().__init__(path)

    def load(self, window_size:int, stride:int, ftrim_sec:int=3, btrim_sec:int=3, subjects:Optional[list]=None) -> Tuple[np.ndarray, np.ndarray]:
        segments, meta = load(path=self.path)
        segments = [m.join(seg) for seg, m in zip(segments, meta)]

        x_frames, y_frames = [], []
        for seg in segments:
            fs = split_using_sliding_window(
                np.array(seg), window_size=window_size, stride=stride,
                ftrim=Sampling_Rate*ftrim_sec, btrim=Sampling_Rate*btrim_sec,
                return_error_value=None)
            if fs is not None:
                x_frames += [fs[:, :, 3:]]
                y_frames += [np.uint8(fs[:, 0, 1:2][..., ::-1])]
            else:
                pass
        x_frames = np.concatenate(x_frames).transpose([0, 2, 1])
        y_frames = np.concatenate(y_frames)
        y_frames -= np.min(y_frames)
        y_frames = y_frames.squeeze(1)

        # subject filtering
        if subjects is not None:
            flags = np.zeros(len(x_frames), dtype=bool)
            for sub in subjects:
                flags = np.logical_or(flags, y_frames[:, 1] == sub)
            x_frames = x_frames[flags]
            y_frames = y_frames[flags]

        return x_frames, y_frames


def load(path:Union[Path,str]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    path = check_path(path)
    raw = load_raw(path)
    data, meta = reformat(raw)
    return data, meta


def load_raw(path:Path) -> pd.DataFrame:
    path = path / 'WISDM_ar_v1.1_raw.txt'
    with path.open('r') as fp:
        whole_str = fp.read()

    whole_str = whole_str.replace(',;', ';')
    semi_separated = re.split('[;\n]', whole_str)
    semi_separated = list(filter(lambda x: x != '', semi_separated))
    comma_separated = [r.strip().split(',') for r in semi_separated]

    # debug
    for s in comma_separated:
        if len(s) != 6:
            print('[miss format?]: {}'.format(s))

    raw_data = pd.DataFrame(comma_separated)
    raw_data.columns = ['user', 'activity', 'timestamp', 'x-acceleration', 'y-acceleration', 'z-acceleration']
    raw_data['z-acceleration'] = raw_data['z-acceleration'].replace('', np.nan)

    # convert activity name to activity id
    raw_data = raw_data.replace(list(ACTIVITIES), list(range(len(ACTIVITIES))))

    raw_data = raw_data.astype({'user': 'uint8', 'activity': 'uint8', 'timestamp': 'uint64', 'x-acceleration': 'float64', 'y-acceleration': 'float64', 'z-acceleration': 'float64'})
    raw_data[['x-acceleration', 'y-acceleration', 'z-acceleration']] = raw_data[['x-acceleration', 'y-acceleration', 'z-acceleration']].fillna(method='ffill')

    return raw_data


def reformat(raw) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    raw_array = raw.to_numpy()
    
    # segment (by user and activity)
    sdata_splited_by_subjects = split_using_target(src=raw_array, target=raw_array[:, 0])
    segments = []
    for sub_id in sdata_splited_by_subjects.keys():
        for src in sdata_splited_by_subjects[sub_id]:
            splited = split_using_target(src=src, target=src[:, 1])
            for act_id in splited.keys():
                segments += splited[act_id]

    segments = list(map(lambda seg: pd.DataFrame(seg, columns=raw.columns).astype(raw.dtypes.to_dict()), segments))
    data = list(map(lambda seg: pd.DataFrame(seg.iloc[:, 3:], columns=raw.columns[3:]), segments))
    meta = list(map(lambda seg: pd.DataFrame(seg.iloc[:, :3], columns=raw.columns[:3]), segments))

    return data, meta 

if __name__ == '__main__':

    output_dir = r'../../data/wisdm'

    wisdm_path = Path('./')
    wisdm = WISDM(wisdm_path)

    x, y = wisdm.load(window_size=256, stride=256, ftrim_sec=0, btrim_sec=0)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    train_sample_num = int(1.0 * len(X_train))  # select x% data
    train_sample_list = [i for i in range(len(X_train))]
    train_sample_list = random.sample(train_sample_list, train_sample_num)
    X_train = X_train[train_sample_list, :]
    y_train = y_train[train_sample_list]

    test_sample_num = int(1.0 * len(X_test))  #  select x% data
    test_sample_list = [i for i in range(len(X_test))]
    test_sample_list = random.sample(test_sample_list, test_sample_num)
    X_test = X_test[test_sample_list, :]
    y_test = y_test[test_sample_list]

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_train)
    dat_dict["labels"] = torch.from_numpy(y_train)
    torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_val)
    dat_dict["labels"] = torch.from_numpy(y_val)
    torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

    dat_dict = dict()
    dat_dict["samples"] = torch.from_numpy(X_test)
    dat_dict["labels"] = torch.from_numpy(y_test)
    torch.save(dat_dict, os.path.join(output_dir, "test.pt"))


