# coding=utf-8
# author=uliontse

import numpy
import pandas
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from mlgb import get_model, mtl_models
from mlgb.data import get_multitask_label_data
from mlgb.utils import check_filepath


if __name__ == '__main__':
    model_name = 'PEPNet'
    seed = 0
    print(f'model_name: {model_name}')

    tmp_dir = '.tmp'
    model_dir = f'{tmp_dir}/{model_name}_tf'
    log_dir = f'{model_dir}/log_dir'
    save_model_dir = f'{model_dir}/save_model'
    check_filepath(tmp_dir, model_dir, log_dir, save_model_dir)

    device = 'cuda' if tf.test.is_built_with_cuda() else 'cpu'
    print(f'device: {device}')

    feature_names, (x_train, y_train), (x_test, y_test) = get_multitask_label_data(
        n_samples=int(32),
        negative_class_weight=0.9,
        multitask_cvr=0.5,
        test_size=0.15,
        seed=seed,
    )
    y_train = (y_train[:, 0], y_train[:, 1])
    y_test = (y_test[:, 0], y_test[:, 1])
    print("fea type distribution: ", [len(names) for names in feature_names])

    model = get_model(
        feature_names=feature_names,
        model_name=model_name,
        task=('binary', 'binary'),
        aim='mtl',
        lang='tf',
        seed=seed,
    )
    model.compile(
        loss=[tf.losses.BinaryCrossentropy(), tf.losses.BinaryCrossentropy()],
        optimizer=tf.optimizers.Nadam(learning_rate=1e-3),
        metrics=[
            tf.metrics.AUC(),
        ],
    )
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=1,
        validation_split=0.15,
        class_weight=None,
    )