import json
import copy
import glob
import torch
import random
import torchaudio
import numpy as np
import pandas as pd
import pickle, pdb, re

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from audiomentations import Compose, AddBackgroundNoise, PolarityInversion, AddGaussianSNR, TimeMask, TimeStretch

import warnings
warnings.filterwarnings("ignore")

'''
NOTE: 
- last update: 2024.08.17
- iemocap 6way classification까지 구현 완료
- iemocap 6way; dataset: iemocap6

- iemocap6 기존에 txt transcript에서 데이터 로드하는 방식 -> csv에서 문장단위로 읽어오는 방식으로 변경. 
- meld 4way, 6way 추가해둠. 

TODO: 
- Test session 체크 ! 
- 4way class 없애기 
- class weight dict로 만들어서 load 하는 방식 
'''

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

def collate_fn(batch):
    # max of 6s of data
    max_audio_len = min(max([b[0].shape[0] for b in batch]), 16000*6)

    data, text_data, speaker_id, taregt, len_data = list(), list(), list(), list(), list()

    for idx in range(len(batch)):
        # append data
        data.append(padding_cropping(batch[idx][0], max_audio_len))
        
        text_data.append(batch[idx][1])
        speaker_id.append(batch[idx][2])
        
        # append len
        if len((batch[idx][0])) >= max_audio_len: len_data.append(torch.tensor(max_audio_len))
        else: len_data.append(torch.tensor(len((batch[idx][0]))))
        
        # append target
        taregt.append(torch.tensor(batch[idx][3]))
    
    data = torch.stack(data, dim=0)
    len_data = torch.stack(len_data, dim=0)
    target = torch.stack(taregt, dim=0)


    # print(f"dataloader: {data.shape, len_data.shape, target.shape}")
    return data, text_data, speaker_id, target, len_data

def padding_cropping(
    input_wav, size
):
    if len(input_wav) > size:
        input_wav = input_wav[:size]
    elif len(input_wav) < size:
        input_wav = torch.nn.ConstantPad1d(padding=(0, size - len(input_wav)), value=0)(input_wav)
    return input_wav

class EmotionDatasetGenerator(Dataset):
    def __init__(
        self,
        data_list:              list,
        noise_list:             list,
        data_len:               int,
        is_train:               bool=False,
        audio_duration:         int=6,
        model_type:             str="rnn",
        apply_guassian_noise:   bool=False,
        dataset:                   str='iemocap'
    ):
        """
        Set dataloader for emotion recognition finetuning.
        :param data_list:       Audio list files
        :param noise_list:      Audio list files
        :param data_len:        Length of input audio file size
        :param is_train:        Flag for dataloader, True for training; False for dev
        :param audio_duration:  Max length for the audio length
        :param model_type:      Type of the model
        """
        self.data_list              = data_list
        self.noise_list             = noise_list
        self.data_len               = data_len
        self.is_train               = is_train
        self.audio_duration         = audio_duration
        self.model_type             = model_type
        self.apply_guassian_noise   = apply_guassian_noise
        self.data                   = dataset 

        self.transform = Compose([
            AddGaussianSNR(min_snr_in_db=10.0, max_snr_in_db=30.0, p=1.0),
            TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=1.0)
        ])
        
    def __len__(self):
        return self.data_len

    def __getitem__(
        self, item
    ):
        # Read original speech in dev
        # data, _ = torchaudio.load(self.data_list[item][3])
        try:
            data, _ = torchaudio.load(self.data_list[item][3])
        except RuntimeError as e:
            print(f"Failed to load audio file: {self.data_list[item][3]}, Error: {e}")
            # Return a dummy tensor to avoid crashing
            exit(1)
            return torch.zeros(1, 16000)

        ### extract text data 
        file_path = self.data_list[item][4]
        
        speaker_identifier = self.data_list[item][0]
        # if self.data in ['iemocap', 'iemocap6'] :    
        #     txt_data = []
        #     with open(file_path, 'r', encoding='utf-8') as file:
        #         lines = file.readlines()
        #         for line in lines:
        #             if line.startswith(speaker_identifier):
        #                 # 발화자 식별자를 제거하고 문장만 추출
        #                 sentence = line.split(':', 1)[1].strip()
        #                 txt_data.append(sentence)
        if self.data in ['iemocap6']: # txt trainscript가 아닌, csv에서 불러옴. 
            files = pd.read_csv(file_path)
            txt_data = files[files['Speaker'] == speaker_identifier]['Utterance'].values.tolist()
        elif self.data == 'meld':
            files = pd.read_csv(file_path)
            txt_data = files[files['Sr No.'] == speaker_identifier]['Utterance'].values.tolist()

        data = data[0]
        if data.isnan()[0].item(): data = torch.zeros(data.shape)
        if len(data) > self.audio_duration*16000: data = data[:self.audio_duration*16000]
        if self.is_train:
            data = data.detach().cpu().numpy()
            data = self.transform(samples=data, sample_rate=16000)
            data = torch.tensor(data)
        if self.data in ['iemocap','iemocap6']:
            return data, txt_data, int(self.data_list[item][5])-1, self.data_list[item][-1]
        else:
            return data, txt_data, int(self.data_list[item][5]), self.data_list[item][-1]

    def _padding_cropping(
        self, input_wav, size
    ):
        if len(input_wav) > size:
            input_wav = input_wav[:size]
        elif len(input_wav) < size:
            input_wav = torch.nn.ConstantPad1d(padding=(0, size - len(input_wav)), value=0)(input_wav)
        return input_wav
    

def include_for_finetune(
    data: list, dataset: str
):
    """
    Return flag for inlusion of finetune.
    :param data:        Input data entries [key, filepath, labels]
    :param dataset:     Input dataset name
    :return: flag:      True to include for finetuning, otherwise exclude for finetuning
    """
    if dataset in ["iemocap", "iemocap_impro"]:
        # IEMOCAP data include 4 emotions, exc->hap
        if data[-1] in ["neu", "sad","fru", "ang" , "exc", "hap"]: return True

    if dataset in ['iemocap6']:
        if data[-1] in ["neu", "sad", "fru", "ang" , "hap", "exc"]: return True
    if dataset == "meld":
        # MELD data include 4 emotions
        #if data[-1] in ["neutral", "sadness", "anger", "joy", "surprise", "fear", "disgust"]: return True
        if data[-1] in ["neutral", "sadness", "anger", "joy"]: return True
    if dataset == "meld6":
        if data[-1] in ["neutral", "sadness", "anger", "joy", "surprise", "fear", "disgust"]: return True
    if dataset == "cmu-mosei": return True
    if dataset == "ravdess": return True
    return False

def map_label(
    data: list, dataset: str
):  
    """
    Return labels for the input data.
    :param data:        Input data entries [key, filepath, labels]
    :param dataset:     Input dataset name
    :return label:      Label index: int
    """
    label_dict = {
        "iemocap6": {"neu": 0, "sad": 1, "fru": 2, "ang": 3,"hap": 4, "exc": 5},
        "iemocap": {"neu": 0, "sad": 1, "fru": 1, "ang": 2, "exc": 3, "hap": 3},
        "iemocap_impro": {"neu": 0, "sad": 1, "ang": 2, "exc": 3, "hap": 3},

        "meld": {"neutral": 0, "sadness": 1,"anger": 2, "joy": 3},
        "meld6": {"neutral": 0, "sadness": 1,  "anger": 2, "joy": 3, "surprise": 4, "fear":5, "disgust":6},
    }
    if dataset in ["iemocap", "iemocap6", "meld", "crema_d", "iemocap_impro"]:
        return label_dict[dataset][data[-1]]
    if dataset in ["cmu-mosei"]:
        # if data[-1] == 0: return 0
        if data[-1] > 0: return 0
        elif data[-1] <= 0: return 1
    if dataset in ["ravdess"]:
        # calm case, merge with neutral
        if data[-1] == 1: return data[-1]-1
        return data[-1]-2
        
def log_dataset_details(
    input_data_list:    list,
    split:              str,
    dataset:            str
):  
    """
    Log the label distribution of the dataset given the split.
    :param input_data_list:     Input data entries [key, filepath, labels]
    :param split:               Splits: train/dev/test
    :param dataset:             Input dataset name
    :return label_stats: stats of the datasets
    """
    label_dict = {
        "iemocap6": {0: "neu", 1: "sad", 2: "fru", 3: "ang", 4: "hap", 5: "exc"},
        "iemocap": {0: "neu", 1: "sad", 2: "ang", 3: "hap"},
        "iemocap_impro": {0: "neu", 1: "sad", 2: "ang", 3: "hap"},
    
        "meld6": {0: "neutral", 1: "sadness", 2: "anger", 3: "joy", 4: "surprise", 5: "fear", 6: "disgust"},
        "meld": {0: "neutral", 1: "sadness", 2: "anger", 3: "joy"},

        "cmu-mosei": {0: "postive", 1: "negative"},
        "ravdess": {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "fearful", 5: "disgust", 6: "surprised"}
    }
    
    label_stats = dict()
    for data in input_data_list:
        # print("input_data_list", input_data_list)
        if data[-1] not in label_stats: label_stats[data[-1]] = 0
        label_stats[data[-1]] += 1
    
    logging.info(f'------------------------------------------------')
    logging.info(f'Number of {split} audio files {dataset}: {len(input_data_list)}')
    for label in label_stats:
        logging.info(f'Number of {split} audio files {label_dict[dataset][label]}: {label_stats[label]}')
    logging.info(f'------------------------------------------------')
    return label_stats
    

def load_pretrain_audios(
    input_path: str
):
    """
    Load pretrain audio data.
    :param input_path: Input data path
    :return train_file_list, dev_file_list: train and dev file list, we don't have test in pretrain
    """
    train_file_list, dev_file_list = list(), list()
    train_stats_dict, dev_stats_dict = dict(), dict()
    for dataset in ['iemocap', 'iemocap6', 'meld',  'ravdess',  'cmu-mosei', ]:
        with open(str(Path(input_path).joinpath(f'{dataset}.json')), "r") as f: 
            split_dict = json.load(f)
        # some stats
        train_stats_dict[dataset] = len(split_dict['train'])
        dev_stats_dict[dataset] = len(split_dict['dev'])

        for split in ['train', 'dev']:
            for data in split_dict[split]:
                if split == 'train': train_file_list.append(data)
                elif split == 'dev': dev_file_list.append(data)

    # logging train file nums
    logging.info(f'------------------------------------------------')
    logging.info(f'Number of train audio files {len(train_file_list)}')
    logging.info(f'------------------------------------------------')
    for dataset in ['iemocap','iemocap6', 'meld', 'meld6', 'ravdess', 'cmu-mosei']:
        logging.info(f'Number of train audio files {dataset}: {train_stats_dict[dataset]}')
    logging.info(f'------------------------------------------------')
    
    # logging dev file nums
    logging.info(f'------------------------------------------------')
    logging.info(f'Number of dev audio files {len(dev_file_list)}')
    logging.info(f'------------------------------------------------')
    for dataset in ['iemocap','iemocap6', 'meld', 'meld6', 'ravdess',  'cmu-mosei']:
        logging.info(f'Number of dev audio files {dataset}: {dev_stats_dict[dataset]}')
    logging.info(f'------------------------------------------------')
    
    return train_file_list, dev_file_list


def load_finetune_audios(
    input_path:     str,
    audio_path:     str,
    dataset:        str,
    fold_idx:       int
):
    """
    Load finetune audio data.
    :param input_path:  Input data path
    :param dataset:     Dataset name
    :param fold_idx:    Fold idx
    :return train_file_list, dev_file_list: train, dev, and test file list
    """
    train_file_list, dev_file_list, test_file_list = list(), list(), list()
    if dataset in ["iemocap_impro"]:
        with open(str(Path(input_path).joinpath(f'iemocap_fold{fold_idx}.json')), "r") as f: split_dict = json.load(f)
    elif dataset in ["crema_d_complete"]:
        with open(str(Path(input_path).joinpath(f'crema_d_fold{fold_idx}.json')), "r") as f: split_dict = json.load(f)
    elif dataset in ["iemocap", "iemocap6","crema_d", "ravdess", "msp-improv"]:
        with open(str(Path(input_path).joinpath(f'{dataset}_fold{fold_idx}.json')), "r") as f: split_dict = json.load(f)
    elif dataset in ["msp-podcast", "meld", "meld6"]: ## add meld
        with open(str(Path(input_path).joinpath(f'{dataset}.json')), "r") as f: split_dict = json.load(f)
    
    for split in ['train', 'dev', 'test']:
        for data in split_dict[split]:
            # pdb.set_trace()
            if include_for_finetune(data, dataset):
                data[-1] = map_label(data, dataset)
                if dataset == "iemocap_impro" and "impro" not in data[0]: continue
                speaker_id, file_path  = data[1], data[3]

                if dataset in ['iemocap', 'iemocap6','msp-improv','crema_d', 'msp-podcast']:
                    output_path = Path(audio_path).joinpath(dataset, file_path.split('/')[-1])
                

                elif dataset in ['ravdess', 'emov_db', 'vox-movie']:
                    output_path = Path(audio_path).joinpath(dataset, f'{speaker_id}_{file_path.split("/")[-1]}')
                #data[3] = str(output_path)
                if split == 'train': train_file_list.append(data)
                elif split == 'dev': dev_file_list.append(data)
                elif split == 'test': test_file_list.append(data)


    # logging train/dev/test file nums
    log_dataset_details(train_file_list, split='train', dataset=dataset)
    log_dataset_details(dev_file_list, split='dev', dataset=dataset)
    log_dataset_details(test_file_list, split='test', dataset=dataset)
    return train_file_list, dev_file_list, test_file_list


def return_weights(
    input_path:     str,
    dataset:        str,
    fold_idx:       int
):
    """
    Return training weights.
    :param input_path:  Input data path
    :param dataset:     Dataset name
    :param fold_idx:    Fold idx
    :return weights:    Class weights
    """
    train_file_list = list()
    if dataset in ["iemocap_impro"]:
        with open(str(Path(input_path).joinpath(f'iemocap_fold{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
    elif dataset in ["crema_d_complete"]:
        with open(str(Path(input_path).joinpath(f'crema_d_fold{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
    elif dataset in ["msp-podcast", "meld"]:
        with open(str(Path(input_path).joinpath(f'{dataset}.json')), "r") as f:
            split_dict = json.load(f) 
    else:
        with open(str(Path(input_path).joinpath(f'{dataset}_fold{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
    
    for data in split_dict['train']:
        if include_for_finetune(data, dataset):
            data[-1] = map_label(data, dataset)
            train_file_list.append(data)
            
    # logging train file nums
    weights_stats = log_dataset_details(train_file_list, split='train', dataset=dataset)
    # compute weight 
    weights = torch.tensor([weights_stats[c] for c in range(len(weights_stats))]).float()
    weights = weights.sum() / weights
    weights = weights / weights.sum()

    return weights

def return_dataset_stats(
    input_path:     str,
    dataset:        str,
    fold_idx:       int
):
    """
    Return training weights.
    :param input_path:  Input data path
    :param dataset:     Dataset name
    :param fold_idx:    Fold idx
    :return weights:    Class weights
    """
    train_file_list = list()
    if dataset in ["iemocap_impro"]:
        with open(str(Path(input_path).joinpath(f'iemocap_fold{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
    elif dataset in ["crema_d_complete"]:
        with open(str(Path(input_path).joinpath(f'crema_d_fold{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
    elif dataset in ["msp-podcast"]:
        with open(str(Path(input_path).joinpath(f'{dataset}.json')), "r") as f:
            split_dict = json.load(f)
    else:
        with open(str(Path(input_path).joinpath(f'{dataset}_fold{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
    
    for split in ["train", "dev", "test"]:
        for data in split_dict[split]:
            if include_for_finetune(data, dataset):
                data[-1] = map_label(data, dataset)
                train_file_list.append(data)
            
    # logging train file nums
    log_dataset_details(train_file_list, split='train', dataset=dataset)

def return_speakers(
    input_file_list:    list
):
    """
    Return training weights.
    :param input_file_list:     input file list
    :return speakers:           unique speakers
    """
    speaker_list = list()
    for input_data in input_file_list: speaker_list.append(input_data[1])
    speaker_list = list(set(speaker_list))
    speaker_list.sort()
    return speaker_list

def set_finetune_dataloader(
    args:                   dict,
    input_file_list:        list,
    is_train:               bool,
    is_distributed:         bool=False,
    rank:                   int=0,
    world_size:             int=2,
    apply_guassian_noise:   bool=False
):
    """
    Return dataloader for finetune experiments.
    :param data:                    Input data entries [key, filepath, labels]
    :param is_train:                Flag for training or not
    :param is_distributed:          Flag for distributed training or not
    :param rank:                    Current GPU rank
    :param world_size:              Total GPU sizes
    :param apply_guassian_noise:    Apply Guassian Noise to audio or not
    :return dataloader:             Dataloader
    """

    # noise files
    noise_wav_files = glob.glob(
        "/media/data/projects/speech-privacy/emo2vec/noise_audio/*.wav"
    )
    
    # dataloader
    filtered_file_list = list()
    for file_path in input_file_list:
        if file_path[3] not in [
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/-7161_hlBOP5NskhM.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/678639_9K5mYSaoBL4.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/607281_9K5mYSaoBL4.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/730042_9K5mYSaoBL4.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/643200_9K5mYSaoBL4.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/570565_9K5mYSaoBL4.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/78720_ULkFbie8g-I.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/0_z7FicxE_pMU.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/7524_-mJ2ud6oKI8.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/78403_P0WaXnH37uI.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/255120_ULkFbie8g-I.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/0_278474.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/77605_bUFAN2TgPaU.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/491385_9bAgEmihzLs.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/96761_-mJ2ud6oKI8.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/133159_TxRS6vJ9ak0.wav",
            "/media/data/projects/speech-privacy/emo2vec/audio/cmu-mosei/train/768515_9K5mYSaoBL4.wav"
        ]:
            filtered_file_list.append(file_path)
    
    data_generator = EmotionDatasetGenerator(
        data_list=filtered_file_list, 
        noise_list=noise_wav_files,
        data_len=len(filtered_file_list),
        is_train=is_train,
        audio_duration = args.max_audio_len,
        model_type=args.downstream_model,
        apply_guassian_noise=apply_guassian_noise,
        dataset=args.dataset
    )

    if is_distributed:
        datasampler = torch.utils.data.distributed.DistributedSampler(
            data_generator, shuffle=True
        )

        dataloader = DataLoader(
            data_generator, 
            batch_size=args.batch_size, 
            num_workers=2, 
            drop_last=True,
            sampler=datasampler
        )
    else:
        if is_train:
            dataloader = DataLoader(
                data_generator, 
                batch_size=args.batch_size, 
                num_workers=args.num_workers, # changed
                shuffle=is_train,
                collate_fn=collate_fn,
                drop_last=is_train
            )
        else:
            dataloader = DataLoader(
                data_generator, 
                batch_size=args.batch_size, 
                num_workers=args.num_workers, # changed
                shuffle=is_train,
                collate_fn=collate_fn,
                drop_last=is_train
            )
    return dataloader
