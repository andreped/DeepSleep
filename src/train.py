import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
from .models import get_model
import numpy as np
from .losses import categorical_focal_loss
from datetime import datetime, date
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import mne
from mne.datasets.sleep_physionet.age import fetch_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


# set mne verbosity
mne.set_log_level(False)

# global vars
annotation_desc_2_event_id = {
            'Sleep stage W': 1,
            'Sleep stage 1': 2,
            'Sleep stage 2': 3,
            'Sleep stage 3': 4,
            'Sleep stage 4': 4,
            'Sleep stage R': 5
        }

# create a new event_id that unifies stages 3 and 4
event_id = {
    'Sleep stage W': 1,
    'Sleep stage 1': 2,
    'Sleep stage 2': 3,
    'Sleep stage 3/4': 4,
    'Sleep stage R': 5
}


def get_epochs_from_raw(raw, annot):
    raw_train = mne.io.read_raw_edf(raw, stim_channel='Event marker', infer_types=True)
    annot_train = mne.read_annotations(annot)
    raw_train.set_annotations(annot_train, emit_warning=False)

    # keep last 30-min wake events before sleep and first 30-min wake events after
    # sleep and redefine annotations on raw data
    annot_train.crop(annot_train[1]['onset'] - 30 * 60,
                     annot_train[-2]['onset'] + 30 * 60)
    raw_train.set_annotations(annot_train, emit_warning=False)

    events_train, _ = mne.events_from_annotations(
        raw_train, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    tmax = 30. - 1. / raw_train.info['sfreq']  # tmax in included

    epochs_train = mne.Epochs(raw=raw_train, events=events_train,
                                event_id=event_id, tmin=0., tmax=tmax, baseline=None)
    
    return epochs_train


def eeg_power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    spectrum = epochs.compute_psd(picks='eeg', fmin=0.5, fmax=30.)
    psds, freqs = spectrum.get_data(return_freqs=True)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)


class Trainer:
    def __init__(self, ret):
        self.ret = ret
        date_time = self.get_datetime()
        self.name = date_time[0] + "_" + date_time[1] + "_classifier_arch_" + ret.arch
        self.history_path = "output/history/"
        self.model_path = "output/models/"
        self.dataset_path = "output/datasets/" + self.name + "/"

    def fit(self):
        subjects = list(range(83))  # 0-82
        missing = [39, 68, 69, 78, 79, 36, 52]
        for m in missing:
            subjects.remove(m)

        all_files = fetch_data(subjects=subjects, recording=[1])

        print("number of subjects:", len(all_files))

        train_files = all_files[:60]  # [:60]
        test_files = all_files[60:]  # [60:]
        
        # Fpz-Cz (EEG) is the variable of interest

        # need to preprocess the training set by looping over all subjects
        train_epoch_list = []
        for raw, annot in tqdm(train_files, "train"):
            try:
                epochs_train = get_epochs_from_raw(raw, annot)
                train_epoch_list.append(epochs_train)
            except ValueError as e:
                print(e)

        train_data = mne.concatenate_epochs(train_epoch_list)

        # prepare test set
        test_epoch_list = []
        for raw, annot in tqdm(test_files, "test"):
            try:
                epoch_test = get_epochs_from_raw(raw, annot)
                test_epoch_list.append(epoch_test)
            except ValueError as e:
                print(e)
        
        test_data = mne.concatenate_epochs(test_epoch_list)

        ## train multiclass model
        pipe = make_pipeline(
            FunctionTransformer(eeg_power_band, validate=False),
            RandomForestClassifier(n_estimators=100, random_state=42)
        )

        # Train
        y_train = train_data.events[:, 2]

        pipe.fit(train_data, y_train)

        # Test
        y_pred = pipe.predict(test_data)

        # Assess the results
        y_test = test_data.events[:, 2]
        acc = accuracy_score(y_test, y_pred)

        print("Accuracy score: {}".format(acc))

        # confusion matrix
        print(confusion_matrix(y_test, y_pred))

        # report
        print(classification_report(y_test, y_pred, target_names=event_id.keys()))
