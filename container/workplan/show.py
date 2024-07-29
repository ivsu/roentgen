import gc
import numpy as np
import pandas as pd
from gluonts.dataset.field_names import FieldName
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def time_series_by_year(data: list[dict]):
    fig = go.Figure()

    data_index = 0

    def get_data(index):
        rows: dict = data[index]['years']
        return list(rows.values()), list(rows.keys())

    targets, years = get_data(data_index)
    channel_names = [mod['label'] for mod in data]

    for i, target in enumerate(targets):
        fig.add_trace(go.Scatter(
            x=np.arange(len(target)) + 1,
            y=target,
            mode='lines',
            # line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
            name=years[i],
        ))

    filter_buttons = []
    for i, channel in enumerate(channel_names):
        targets, years = get_data(i)
        filter_buttons.append(dict(
            args=[dict(
                y=targets,
                # selector=dict(name='Прогноз'),
                ),
                dict(title=f'Количество исследований по годам по модальности: {channel}'),
                np.arange(len(targets))
            ],
            label=channel,
            method='update'
        ))

    fig.update_xaxes(title_text="Недели года")
    fig.update_yaxes(title_text="Количество исследований")
    fig.update_layout(
        title_text=f'Количество исследований по годам по модальности: {channel_names[data_index]}',
        # autosize=False, width=700, height=500,
    )
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=filter_buttons,
                direction="down",
            ),
        ],
    )
    fig.show()


def dashboard(metrics, dataset, forecasts, hp,
              channel_names, total_periods, name):
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Функция ошибки', 'MASE/sMAPE',
                        # f'Прогноз/факт [{channel_names[ts_index]}]')
                        f'Прогноз/факт [...]')
    )

    losses = np.asarray(metrics['losses'])  # axis 0: epochs, axis 1: batches
    loss_mean = losses.mean(axis=1)
    loss_std = losses.std(axis=1)
    epochs = np.arange(losses.shape[0])

    mase, smape = metrics['mase'], metrics['smape']

    freq = hp.get('freq')
    prediction_len = hp.get('prediction_len')

    ts_index = 0
    ds = dataset[ts_index]
    forecast = forecasts[ts_index]

    index = pd.period_range(
        start=dataset[0][FieldName.START],
        periods=len(dataset[0][FieldName.TARGET]),
        freq=freq,
    ).to_timestamp()

    # loss

    # стандартное отклонение ±1
    fig.add_trace(
        go.Scatter(
            x=epochs, y=loss_mean + loss_std,
            mode='lines',
            line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
            showlegend=False,
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=epochs, y=loss_mean - loss_std,
            mode='lines',
            line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
            # заполняем область относительно предыдущей линии
            fill='tonexty', fillcolor='rgba(192, 192, 192, 0.3)',
            name="± 1-std",
            showlegend=False,
        ),
        row=1, col=1
    )
    # среднее значение ошибки на эпоху
    fig.add_trace(
        go.Scatter(
            x=epochs, y=loss_mean,
            name="Ошибка"
        ),
        row=1, col=1
    )

    # MASE/sMAPE

    fig.add_trace(
        go.Scatter(
            x=mase, y=smape,
            text=channel_names, textposition="top center",
            mode='markers+text',
            # marker_size=[40, 60, 80, 100]
            name="Метрики",
        ),
        row=1, col=2
    )

    # План / Факт

    # стандартное отклонение ±1
    fig.add_trace(
        go.Scatter(
            x=index[-prediction_len:],
            y=forecast.mean(axis=0) + forecast.std(axis=0),
            mode='lines',
            line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
            showlegend=False,
        ),
        row=1, col=3
    )
    fig.add_trace(
        go.Scatter(
            x=index[-prediction_len:],
            y=forecast.mean(axis=0) - forecast.std(axis=0),
            mode='lines',
            line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
            fill='tonexty', fillcolor='rgba(192, 192, 192, 0.3)',
            name="± 1-std",
        ),
        row=1, col=3
    )
    # медиана прогноза
    fig.add_trace(
        go.Scatter(
            x=index[-prediction_len:],
            y=np.median(forecast, axis=0),
            line=dict(color='DodgerBlue'),
            name="Прогноз"
        ),
        row=1, col=3
    )
    # истинное значение
    fig.add_trace(
        go.Scatter(
            x=index[-total_periods * prediction_len:],
            y=ds["target"][-total_periods * prediction_len:],
            line=dict(color='lightsalmon'),
            name="Факт"
        ),
        row=1, col=3
    )

    filter_buttons = []
    for i, channel in enumerate(channel_names):
        filter_buttons.append(dict(
            args=[dict(
                y=[
                    forecasts[i].mean(axis=0) + forecasts[i].std(axis=0),
                    forecasts[i].mean(axis=0) - forecasts[i].std(axis=0),
                    np.median(forecasts[i], axis=0),
                    dataset[i]["target"][-total_periods * prediction_len:],
                ],
                # selector=dict(name='Прогноз'),
                ),
                # dict(subplot_titles=('Функция ошибки', 'MASE/sMAPE',
                #                      f'Прогноз/факт [{channel_names[i]}]')),
                [4, 5, 6, 7]
            ],
            label=channel,
            method='update'
        ))

    fig.update_layout(
        title_text=f'Бот {name}',
        # autosize=False, width=700, height=500,
        updatemenus=[
            dict(
                buttons=filter_buttons,
                direction="down",
            ),
        ],
    )

    fig.update_xaxes(title_text="Эпоха", row=1, col=1)
    fig.update_yaxes(title_text="Negative Log Likelihood (NLL)", title_standoff=0, row=1, col=1)

    fig.update_xaxes(title_text="MASE", row=1, col=2)
    fig.update_yaxes(title_text="sMAPE", title_standoff=0, row=1, col=2)

    fig.update_xaxes(title_text="Период: " + freq, row=1, col=3)
    fig.update_yaxes(title_text="Количество исследований", title_standoff=0, row=1, col=3)

    fig.show()


def indicators(hp, params, titles):
    fig = go.Figure()
    for i, p in enumerate(params):
        fig.add_trace(go.Indicator(
            mode="number",
            value=hp.get(p),
            title={'text': titles[i]},
            domain={'row': 0, 'column': i}))
    fig.update_layout(
        # title_text='Запуск обучения ботов',
        # width=500,
        autosize=False, height=250,
    )
    fig.update_layout(
        grid={'rows': 1, 'columns': len(params), 'pattern': "independent"},
        )
    fig.show()
    # fig = None
    # gc.collect()


def test(figure=None, row=1, col=1):

    fig = figure if figure is not None else go.Figure()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['Тест', 'empty']
    )

    gen = np.random.default_rng()
    data = gen.integers(0, 100, size=(100,))
    x = np.arange(data.shape[0])

    fig.add_trace(
        go.Scatter(
            x=x, y=data,
            mode='lines',
            showlegend=True,
            name='random'
        ),
        row=row, col=col
    )
    # fig.layout.annotations[0].update(text="Stackoverflow")
    fig.show()
