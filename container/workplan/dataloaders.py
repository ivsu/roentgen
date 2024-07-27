import os
import torch
# для трансформации и обогащения данных
from gluonts.time_feature import (
    # функции фреймворка GluonTS для формирования дополнительных признаков временного ряда
    time_features_from_frequency_str,
)
from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)
# для нарезки временного ряда на окна
from gluonts.transform.sampler import InstanceSampler
from typing import Optional

# базовый класс для определения трансформации
from transformers import PretrainedConfig
# для загрузчиков данных
from typing import Iterable
from gluonts.itertools import Cached, Cyclic
from gluonts.dataset.loader import as_stacked_batches


def create_transformation(freq: str, config: PretrainedConfig, time_features: list) -> Transformation:
    # определяем удаляемые поля
    remove_field_names = []
    if config.num_static_real_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_REAL)
    if config.num_dynamic_real_features == 0:
        remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
    if config.num_static_categorical_features == 0:
        remove_field_names.append(FieldName.FEAT_STATIC_CAT)

    # объединим и вернём все трансформации
    return Chain(
        # Шаг 1: удаляем статические/динамические поля при их отсутствии
        [RemoveFields(field_names=remove_field_names)]
        # Шаг 2: конвертируем данные в NumPy (потенциально это не нужно)
        + ([AsNumpyArray(
            field=FieldName.FEAT_STATIC_CAT,
            expected_ndim=1,
            dtype=int,
        )]
           if config.num_static_categorical_features > 0
           else [])
        + ([AsNumpyArray(
            field=FieldName.FEAT_STATIC_REAL,
            expected_ndim=1,
        )]
           if config.num_static_real_features > 0
           else [])
        + [AsNumpyArray(
            field=FieldName.TARGET,
            # дополнительное измерение в случае многомерности (Multivariate):
            expected_ndim=1 if config.input_size == 1 else 2
            ),
            # Шаг 3: заполняем значения NaN в target нулями и возвращаем маску
            # для наблюдаемых значений (true для наблюдаемых значений, false для Nan);
            # декодер использует эту маску, исключая расчёт потерь на ненаблюдаемых значениях;
            # (см. loss_weights в модели xxxForPrediction)
            AddObservedValuesIndicator(
                target_field=FieldName.TARGET,
                output_field=FieldName.OBSERVED_VALUES,
            ),
            # Шаг 4: добавляем временные признаки, основываясь на частоте данных в датасете:
            # день недели, день месяца, день года для freq='1D';
            # это работает как позиционное кодирование;
            AddTimeFeatures(
                start_field=FieldName.START,
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_TIME,
                # time_features=time_features_from_frequency_str(freq),
                time_features=time_features,
                pred_length=config.prediction_length,
            ),
            # Шаг 5: добавляем дополнительный временной признак - возраст,
            # сообщающий модели, где на временной шкале находится значение временного ряда
            # (что-то вроде счётчика); временная шкала логарифмическая;
            AddAgeFeature(
                target_field=FieldName.TARGET,
                output_field=FieldName.FEAT_AGE,
                pred_length=config.prediction_length,
                log_scale=True,
            ),
            # Шаг 6: вертикально стыкуем все временные признаки в ключ FEAT_TIME,
            # также добавляя динамические признаки
            VstackFeatures(
                output_field=FieldName.FEAT_TIME,
                input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                             + ([FieldName.FEAT_DYNAMIC_REAL]
                                if config.num_dynamic_real_features > 0
                                else [])
            ),
            # Шаг 7: переименовываем все поля для соответствия именам HuggingFace
            RenameFields(
                mapping={
                    FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                    FieldName.FEAT_STATIC_REAL: "static_real_features",
                    FieldName.FEAT_TIME: "time_features",
                    FieldName.TARGET: "values",
                    FieldName.OBSERVED_VALUES: "observed_mask",
                }
            ),
        ]
    )


def create_instance_splitter(
        config: PretrainedConfig,
        mode: str,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
) -> Transformation:
    """
    Нарезает данные на последовательности
    """
    assert mode in ["train", "validation", "test"]

    instance_sampler = {
        "train": train_sampler or ExpectedNumInstanceSampler(
            num_instances=1.0, min_future=config.prediction_length
        ),
        "validation": validation_sampler or ValidationSplitSampler(
            min_future=config.prediction_length
        ),
        "test": TestSplitSampler(),
    }[mode]

    return InstanceSplitter(
        target_field="values",
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=config.context_length + max(config.lags_sequence),
        future_length=config.prediction_length,
        time_series_fields=["time_features", "observed_mask"],
    )


def create_train_dataloader(
        config: PretrainedConfig,
        freq,
        data,
        batch_size: int,
        num_batches_per_epoch: int,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = True,
        time_features: list = None,
        **kwargs,
) -> Iterable:
    """ Загружает, трансформирует и нарезает на батчи тренировочную выборку.
    """
    prediction_input_names = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        prediction_input_names.append("static_categorical_features")

    if config.num_static_real_features > 0:
        prediction_input_names.append("static_real_features")

    training_input_names = prediction_input_names + [
        "future_values",
        "future_observed_mask",
    ]

    transformation = create_transformation(freq, config, time_features)
    transformed_data = transformation.apply(data, is_train=True)
    if cache_data:
        transformed_data = Cached(transformed_data)

    # создаём обучающий инстанс
    instance_splitter = create_instance_splitter(config, "train")

    # сплиттер будет рандомно сэмплировать из временного ряда окно длиной:
    # context length + lags + prediction length (из 4-х возможных временных рядов);
    # возвращает итератор
    stream = Cyclic(transformed_data).stream()
    training_instances = instance_splitter.apply(
        stream, is_train=True
    )
    # print(f'batch_size: {batch_size}, num_batches_per_epoch: {num_batches_per_epoch}')

    return as_stacked_batches(
        training_instances,
        batch_size=batch_size,
        shuffle_buffer_length=shuffle_buffer_length,
        field_names=training_input_names,
        output_type=torch.tensor,
        num_batches_per_epoch=num_batches_per_epoch,
    )


def create_test_dataloader(
        config: PretrainedConfig,
        freq,
        data,
        batch_size: int,
        time_features: list,
        **kwargs,
):
    """ Загружает, трансформирует и нарезает на батчи тестовую выборку.
    """
    prediction_input_names = [
        "past_time_features",
        "past_values",
        "past_observed_mask",
        "future_time_features",
    ]
    if config.num_static_categorical_features > 0:
        prediction_input_names.append("static_categorical_features")

    if config.num_static_real_features > 0:
        prediction_input_names.append("static_real_features")

    transformation = create_transformation(freq, config, time_features)
    transformed_data = transformation.apply(data, is_train=False)

    # создаём сплиттер тестового инстанса, который будет сэмплировать последнее контекстное окно,
    # видимое во время обучения только энкодеру
    instance_sampler = create_instance_splitter(config, "test")

    # применяем трансформацию в режиме тестирования
    testing_instances = instance_sampler.apply(transformed_data, is_train=False)

    return as_stacked_batches(
        testing_instances,
        batch_size=batch_size,
        output_type=torch.tensor,
        field_names=prediction_input_names,
    )
