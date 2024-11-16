import gc
import numpy as np
import pandas as pd
import copy
from gluonts.dataset.field_names import FieldName
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from workplan.datamanager import get_channels_settings


def plot_smth():
    fig = go.Figure(
        data=[go.Table(
            header=dict(
                values=['header'],
                height=30,
                # fill_color='white'
            ),
            cells=dict(
                values=['something'],
                height=30,
            )
        )])
    fig.update_layout(
        title_text='Заголовок',
        width=300,
        height=300,
        paper_bgcolor='rgba(222, 222, 222, 1)',
    )
    fig.show()


def time_series_by_year_v2(data: list[dict]):
    plot_smth()


def time_series_by_year(data: list[dict]):
    plot_smth()
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


def dashboard(metrics, dataset, forecasts, learning_rates, bot, name):
    metrics = copy.deepcopy(metrics)
    learning_rates = copy.deepcopy(learning_rates)
    forecasts = copy.deepcopy(forecasts)
    channels, _, _ = get_channels_settings()

    freq = bot.get('freq')
    prediction_len = bot.get('prediction_len')
    forecast_periods = forecasts.shape[2]
    forecast_stages = forecast_periods // prediction_len
    total_stages = forecast_stages + 1

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Функция ошибки / LR', 'MASE / sMAPE',
                        # f'Прогноз/факт [{channels[ts_index]}]')
                        f'Прогноз/факт [...]'),
        horizontal_spacing=0.08,
        specs=[[{'secondary_y': True}, {'secondary_y': False}, {'secondary_y': False}]]
    )

    def calc_loss(losses: list):
        """Учитывая, что длина последовательности на каждой стадии разная,
        рассчитаем среднее и стандартное отклонение особым образом:

        :param losses: входной список размерности [stages, epochs, batches]
        :return: mean - среднее по всем stages и batches для каждой epochs;
                 std - стандартное отклонение по stages среднего по batches для каждой epochs
        """
        mean = []
        stage_losses = []
        first_run = True
        for stage, stage_data in enumerate(losses):
            stage_losses.append([])
            for epoch, epoch_data in enumerate(stage_data):
                if first_run:
                    mean.append([])
                mean[epoch] += epoch_data
                stage_losses[stage].append(np.array(epoch_data).mean())
            first_run = False

        stage_losses = np.array(stage_losses)
        mean = np.array(mean).mean(axis=1)
        std = stage_losses.std(axis=0)
        return mean, std, stage_losses

    # losses = np.asarray(metrics['losses'])  # [stages, epochs, batches]
    # loss_mean = losses.mean(axis=(0, 2))
    # loss_std = losses.mean(axis=2).std(axis=0)
    loss_mean, loss_std, stage_losses = calc_loss(metrics['losses'])  # [stages, epochs, batches]
    epochs = np.arange(loss_mean.shape[0])

    mase = np.array(metrics['mase']).mean(axis=0)  # [stages, channels]
    smape = np.array(metrics['smape']).mean(axis=0)

    ts_index = 0

    index = pd.period_range(
        start=dataset[0][FieldName.START],
        periods=len(dataset[0][FieldName.TARGET]),
        freq=freq,
    ).to_timestamp()

    trace_index = 0

    # loss

    # стандартное отклонение ±1
    fig.add_trace(
        go.Scatter(
            x=epochs, y=loss_mean + loss_std,
            mode='lines',
            line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
            showlegend=False,
        ),
        row=1, col=1, secondary_y=False,
    )
    trace_index += 1
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
        row=1, col=1, secondary_y=False,
    )
    trace_index += 1
    # среднее значение ошибки на эпоху
    fig.add_trace(
        go.Scatter(
            x=epochs, y=loss_mean,
            mode='lines+markers',
            line=dict(color='forestgreen'),
            name="Ошибка"
        ),
        row=1, col=1, secondary_y=False,
    )
    trace_index += 1
    # learning rates
    fig.add_trace(
        go.Scatter(
            x=epochs, y=learning_rates,
            mode='lines',
            line=dict(width=0.8, color='rgba(178, 34, 34, 0.5)'),  # FireBrick
            name="Learning rate"
        ),
        row=1, col=1, secondary_y=True,
    )
    trace_index += 1

    # дополнительно отрисуем функции ошибок по каждой стадии
    stages = stage_losses.shape[0]
    if stages > 1:
        color_step = int((139 - 34) / (stages - 1))
        for stage in range(stages):
            # среднее значение ошибки на эпоху
            loss_mean = stage_losses[stage]
            fig.add_trace(
                go.Scatter(
                    x=epochs, y=loss_mean,
                    name=f'Этап {stage + 1}',
                    mode='lines',
                    line=dict(
                        width=0.7,
                        color=f'rgba({min(34 + stage * color_step, 255)}, 139, {min(34 + stage * color_step, 255)}, 0.7)'
                    ),  # forestgreen 34, 139, 34
                    # showlegend=stage == 0,
                ),
                row=1, col=1
            )
            trace_index += 1

    # MASE/sMAPE

    fig.add_trace(
        go.Scatter(
            x=mase, y=smape,
            text=channels, textposition="top center",
            mode='markers+text',
            # marker_size=[40, 60, 80, 100]
            name="Метрики",
        ),
        row=1, col=2
    )
    trace_index += 1

    # План / Факт
    previous_fact_slice = slice(-total_stages * prediction_len, -forecast_stages * prediction_len + 1)
    check_fact_slice = slice(-forecast_stages * prediction_len, None)

    planfact = [dict(
        std_plus=forecasts[i].mean(axis=0) + forecasts[i].std(axis=0),
        std_minus=forecasts[i].mean(axis=0) - forecasts[i].std(axis=0),
        plan=np.median(forecasts[i], axis=0),
        # fact=dataset[i]["target"][-total_stages * prediction_len:],
        previous_fact=dataset[i]["target"][previous_fact_slice],
        check_fact=dataset[i]["target"][check_fact_slice],
        fact_1y_ago=dataset[i]["target"][-total_stages * prediction_len - 52:-52],
        fact_2y_ago=dataset[i]["target"][-total_stages * prediction_len - 104:-104],
    ) for i, _ in enumerate(channels)]

    # print(f'index:\n{index[-forecast_periods:]}')
    # print(f'plan:\n{planfact[0]["plan"]}')

    button_start_index = trace_index

    # стандартное отклонение ±1
    fig.add_trace(
        go.Scatter(
            x=index[-forecast_periods:],
            y=planfact[ts_index]['std_plus'],
            mode='lines',
            line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
            showlegend=False,
        ),
        row=1, col=3
    )
    trace_index += 1
    fig.add_trace(
        go.Scatter(
            x=index[-forecast_periods:],
            y=planfact[ts_index]['std_minus'],
            mode='lines',
            line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
            fill='tonexty', fillcolor='rgba(192, 192, 192, 0.3)',
            name="± 1-std",
        ),
        row=1, col=3
    )
    trace_index += 1
    # медиана прогноза
    fig.add_trace(
        go.Scatter(
            x=index[-forecast_periods:],
            y=planfact[ts_index]['plan'],
            mode='lines+markers',
            line=dict(color='DodgerBlue'),
            name="Прогноз"
        ),
        row=1, col=3
    )
    trace_index += 1

    # истинное значение до начала прогноза
    fig.add_trace(
        go.Scatter(
            x=index[previous_fact_slice],
            y=planfact[ts_index]['previous_fact'],
            mode='lines+markers',
            line=dict(color='rgba(255, 160, 122, 1.0)'),  # lightsalmon: 255, 160, 122
            name="Предш. факт"
        ),
        row=1, col=3
    )
    trace_index += 1
    # истинное значение в сравнении с прогнозом
    fig.add_trace(
        go.Scatter(
            x=index[check_fact_slice],
            y=planfact[ts_index]['check_fact'],
            mode='lines+markers',
            # line=dict(color='lightsalmon'),
            line=dict(color='rgba(130, 90, 74, 0.5)'),  # комплиментарный тёмный lightsalmon: 255, 160, 122
            name="Провероч. факт"
        ),
        row=1, col=3
    )
    trace_index += 1

    # истинное значение -52 недели назад
    fig.add_trace(
        go.Scatter(
            x=index[-total_stages * prediction_len:],
            y=planfact[ts_index]['fact_1y_ago'],
            mode='lines',
            # line=dict(width=0.7, color='rgba(250, 128, 114, 0.75)'),  # lightsalmon: rgba(255, 160, 122)
            line=dict(width=0.8, color='rgba(255, 160, 122, 1.0)'),  # lightsalmon: rgba(255, 160, 122)
            name="Факт -52 нед."
        ),
        row=1, col=3
    )
    trace_index += 1
    # истинное значение -104 недели назад
    y = planfact[ts_index]['fact_2y_ago']
    x = index[-total_stages * prediction_len:][-len(y):]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='lines',
            # line=dict(width=0.5, color='rgba(250, 128, 114, 0.5)'),
            line=dict(width=0.6, color='rgba(255, 160, 122, 0.8)'),
            name="Факт -104 нед."
        ),
        row=1, col=3
    )
    trace_index += 1

    filter_buttons = []
    for i, channel in enumerate(channels):
        filter_buttons.append(dict(
            args=[dict(
                y=[v for v in planfact[i].values()],
                # selector=dict(name='Прогноз'),
            ),
                # low_annotations = [dict(x="2015-05-01",
                #                     y=df.Low.mean(),
                #                     xref="x", yref="y",
                #                     text="Low Average:<br> %.2f" % df.Low.mean(),
                #                     ax=-40, ay=40),
                # dict(layout={'title': {'text': 'Title 2'}}),
                {
                    # 'annotations': [dict(text='My text', row=1, col=3),
                                 # dict(text='My text 2'),
                                 # dict(text='My text 3')
                                 # ],
                    # "xaxis": {"title": "Text 1"},
                    # "xaxis1": {"title": "Text 1"},
                    # "xaxis2": {"title": "Text 2"},
                    # "xaxis3": {"title": "Text 3"},
                },
                # {'layout':  dict(subplot_titles=['Plot 1', 'Plot 2', 'Plot 3'])},
                # dict(layout={'title_text': 'My Text', 'row': 1, 'col': 3}),
                # dict(layout={'annotations': [{'title': {'text': 'Stackoverflow'}}]}),
                # layout.annotations[0].update(text="Stackoverflow")
                # dict(subplot_titles=('Функция ошибки', 'MASE/sMAPE',
                #                      f'Прогноз/факт [{CHANNELS[i]}]')),
                [button_start_index + i for i in range(trace_index - button_start_index)]
            ],
            label=channel,
            method='update'
        ))

    #
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

    fig.update_xaxes(title_text="Эпоха", row=1, col=1, tickfont={'size': 11})
    fig.update_yaxes(title_text="Negative Log Likelihood (NLL)", row=1, col=1, secondary_y=False,
                     tickfont={'size': 11})
    fig.update_yaxes(title_text="Learning Rate", row=1, col=1, secondary_y=True,
                     tickformat='4.0e', tickfont={'size': 11})

    fig.update_xaxes(title_text="MASE", row=1, col=2, tickfont={'size': 11})
    fig.update_yaxes(title_text="sMAPE", row=1, col=2, tickfont={'size': 11})

    fig.update_xaxes(title_text="Период: " + freq, row=1, col=3, tickfont={'size': 11})
    fig.update_yaxes(title_text="Количество исследований", row=1, col=3, tickfont={'size': 11})

    # common
    fig.update_yaxes(title_font=dict(size=11), title_standoff=5)
    fig.update_xaxes(title_font=dict(size=11))

    fig.show()


def indicators(hp, params):

    n_in_row = 5
    n_indicators = len(params)
    n_rows = 1 + (n_indicators - 1) // n_in_row
    n_cols = 1 + (n_indicators - 1) // n_rows

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        # horizontal_spacing=0.1,
        specs=[[{"type": "indicator"} for _ in range(n_cols)] for _ in range(n_rows)],
    )

    for i, key in enumerate(params):
        value = hp.get(key)
        fig.add_trace(go.Indicator(
            mode="number",
            value=len(value) if isinstance(value, list) else value,
            number={'font': {'size': 40}},
            title={'text': params[key], 'font': {'size': 10}},
            # domain={'row': 0, 'column': i}
            ),
            row=1 + i // n_in_row, col=1 + i % n_in_row,
        )

    fig.update_layout(
        title_text='Запуск подбора гиперпараметров',
        # width=500,
        # autosize=False,
        height=180 * n_rows,  # это примерно (для точности нужно учесть высоту заголовка)
        # grid={'rows': 1, 'columns': len(params), 'pattern': 'independent'},
        paper_bgcolor='rgba(222, 222, 222, 1)',
        # plot_bgcolor='rgba(128, 128, 192, 1)',  # для индикаторов, похоже, не работает
        # margin={'autoexpand': False, 'pad': 30, 'l': 20}
        # margin=dict(l=30, r=30, t=30, b=20, pad=40),
    )
    fig.show()


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
    fig.show()
