import gc
import numpy as np
import pandas as pd
from gluonts.dataset.field_names import FieldName
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class Show:

    @staticmethod
    def prices_lines(
        train_ds,
        valid_ds,
        test_ds,
        start_index=0  # индекс, с которого отображать все выборки
    ):
        """ Выводит выборки курсов акции """
        fig = go.Figure()
        x = np.arange(len(test_ds[0]['target']))[start_index:]

        # open: index 0
        channel = 0

        # high: index 1
        fig.add_trace(go.Scatter(
            x=x,
            y=test_ds[1]['target'][start_index:],
            mode='lines',
            line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
            name='high',
        ))
        # low: index 2
        fig.add_trace(go.Scatter(
            x=x,
            y=test_ds[2]['target'][start_index:],
            mode='lines',
            line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
            # fill='tonexty', fillcolor='rgba(192, 192, 192, 0.3)',
            name="low",
        ))
        # отрисовка
        fig.add_trace(
            go.Scatter(x=x, y=test_ds[0]['target'][start_index:], name='test')
        )
        fig.add_trace(
            go.Scatter(x=x, y=valid_ds[0]['target'][start_index:], name='valid')
        )
        fig.add_trace(
            go.Scatter(x=x, y=train_ds[0]['target'][start_index:], name='train')
        )
        fig.update_xaxes(title_text="Индекс данных")
        fig.update_yaxes(title_text="Курс акции")
        fig.update_layout(
            title_text='Наложенные друг на друга выборки и волатильность (low, high)',
            autosize=False, width=700, height=500,
        )
        fig.show()

    @staticmethod
    def prices(
        samples,  # список выборок
        start_index=0  # индекс, с которого отображать все выборки
    ):
        valid_ds = None
        if len(samples) == 2:
            train_ds, test_ds = samples
        else:
            train_ds, valid_ds, test_ds = samples
        """ Выводит выборки курсов акции """
        fig = go.Figure()
        x = np.arange(len(test_ds[0]['target']))[start_index:]

        fig.add_trace(
            go.Candlestick(
                x=x,
                open=test_ds[0]['target'][start_index:],
                high=test_ds[1]['target'][start_index:],
                low=test_ds[2]['target'][start_index:],
                close=test_ds[3]['target'][start_index:],
                increasing_line_color='gray', decreasing_line_color='lightgray',
                name='test',
            )
        )
        if valid_ds:
            fig.add_trace(
                go.Candlestick(
                    x=x,
                    open=valid_ds[0]['target'][start_index:],
                    high=valid_ds[1]['target'][start_index:],
                    low=valid_ds[2]['target'][start_index:],
                    close=valid_ds[3]['target'][start_index:],
                    increasing_line_color='mediumblue', decreasing_line_color='deepskyblue',
                    name='valid',
                )
            )
        fig.add_trace(
            go.Candlestick(
                x=x,
                open=train_ds[0]['target'][start_index:],
                high=train_ds[1]['target'][start_index:],
                low=train_ds[2]['target'][start_index:],
                close=train_ds[3]['target'][start_index:],
                name='train',
            )
        )
        fig.update_xaxes(
            title_text="Индекс данных",
            rangeslider=dict(visible=False),
        )
        fig.update_yaxes(title_text="Курс акции")
        fig.update_layout(
            title_text='Наложенные друг на друга выборки',
            # autosize=False, width=700, height=500,
        )
        fig.show()

    @staticmethod
    def losses(data: list, figure=None, row=1, col=1):
        """ Выводит график функции потерь и стандартное отклонение по эпохе"""
        fig = figure if figure else go.Figure()
        # print(f'{"Получено" if figure else "Создано"}')

        data = np.asarray(data)  # axis 0: epochs, axis 1: batches
        x = np.arange(data.shape[0])

        # стандартное отклонение ±1
        fig.add_trace(
            go.Scatter(
                x=x, y=data.mean(axis=1) + data.std(axis=1),
                mode='lines',
                line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
                showlegend=False,
            ),
            row=row, col=col
        )
        fig.add_trace(
            go.Scatter(
                x=x, y=data.mean(axis=1) - data.std(axis=1),
                mode='lines',
                line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
                # заполняем область относительно предыдущей линии
                fill='tonexty', fillcolor='rgba(192, 192, 192, 0.3)',
                name="± 1-std",
                showlegend=False,
            ),
            row=row, col=col
        )

        # среднее значение ошибки на эпоху
        fig.add_trace(
            go.Scatter(
                x=x, y=data.mean(axis=1),
                name="Ошибка"
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Эпоха", row=row, col=col)
        fig.update_yaxes(title_text="Negative Log Likelihood (NLL)", title_standoff=0, row=row, col=col)

        # если отдельный вызов метода
        if figure is None:
            fig.update_layout(
                title_text='Ошибка модели',
                autosize=False, width=700, height=500,
            )
            fig.show()

    @staticmethod
    def estimation(test_ds, forecasts, hp, ts_index,
                   total_periods=2, figure=None, row=1, col=1):
        """
        Выводит графики факта и прогноза,
        а также стандартное отклонение в прогнозных значениях
        """
        fig = figure if figure else go.Figure()

        freq = hp.get('freq')
        prediction_len = hp.get('prediction_len')

        ds = test_ds[ts_index]
        forecast = forecasts[ts_index]

        index = pd.period_range(
            start=ds[FieldName.START],
            periods=len(ds[FieldName.TARGET]),
            freq=freq,
        ).to_timestamp()

        # стандартное отклонение +1
        fig.add_trace(
            go.Scatter(
                x=index[-prediction_len:],
                y=forecast.mean(0) + forecast.std(axis=0),
                mode='lines',
                line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
                showlegend=False,
            ),
            row=row, col=col
        )

        # стандартное отклонение -1
        fig.add_trace(
            go.Scatter(
                x=index[-prediction_len:],
                y=forecast.mean(0) - forecast.std(axis=0),
                mode='lines',
                line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
                fill='tonexty', fillcolor='rgba(192, 192, 192, 0.3)',
                name="± 1-std",
            ),
            row=row, col=col
        )

        # истинное значение
        fig.add_trace(
            go.Scatter(
                x=index[-total_periods * prediction_len:],
                y=ds["target"][-total_periods * prediction_len:],
                name="Факт"
            ),
            row=row, col=col
        )

        # медиана прогноза
        fig.add_trace(
            go.Scatter(
                x=index[-prediction_len:],
                y=np.median(forecast, axis=0),
                line=dict(color='DodgerBlue'),
                name="Прогноз"
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Дата: " + freq, row=row, col=col)
        fig.update_yaxes(title_text="Курс акции", title_standoff=0, row=row, col=col)
        if figure is None:
            fig.update_layout(
                title_text='Сравнение прогноза с фактом',
                # autosize=False, width=700, height=500,
            )
            fig.show()

    @staticmethod
    def metrics(mase_metrics, smape_metrics, channel_names=None, figure=None, row=1, col=1):
        """Выводит метрики ошибок MASE/sMAPE"""
        fig = figure if figure else go.Figure()
        if channel_names:
            fig.add_trace(
                go.Scatter(
                    x=mase_metrics, y=smape_metrics,
                    text=channel_names, textposition="top center",
                    mode='markers+text',
                    # marker_size=[40, 60, 80, 100]
                    name="Метрики",
                ),
                row=row, col=col
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=mase_metrics, y=smape_metrics,
                    mode='markers',
                    name="Метрики",
                ),
                row=row, col=col
            )
        fig.update_xaxes(title_text="MASE", row=row, col=col)
        fig.update_yaxes(title_text="sMAPE", title_standoff=0, row=row, col=col)

        # если отдельный вызов метода
        if figure is None:
            fig.update_layout(
                title_text='Метрики ошибок MASE/sMAPE',
                autosize=False, width=600, height=400,
            )
            fig.show()

    @staticmethod
    def statistic(metrics, dataset, forecasts, hp,
                  channel_names, ts_index, total_periods, name):

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Функция ошибки', 'MASE/sMAPE',
                            f'Прогноз/факт [{channel_names[ts_index]}]')
        )
        Show.losses(metrics['losses'], figure=fig, row=1, col=1)
        Show.metrics(metrics['mase'], metrics['smape'],
                     channel_names, figure=fig, row=1, col=2)
        Show.estimation(dataset, forecasts, hp, ts_index, total_periods,
                        figure=fig, row=1, col=3)
        fig.update_layout(
            title_text=f'Бот {name}',
            # autosize=False, width=700, height=500,
        )
        # fig = self.test(figure=fig, row=1, col=1)
        fig.show()

    @staticmethod
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
        fig = None
        gc.collect()

    def test(self, figure=None, row=1, col=1):

        fig = figure if figure is not None else go.Figure()

        gen = np.random.default_rng()
        data = gen.integers(0, 100, size=(100,))
        x = np.arange(data.shape[0])

        fig.add_trace(
            go.Scatter(
                x=x, y=data,
                mode='lines',
                line=dict(width=0.5, color='rgba(192, 192, 192, 1)'),
                showlegend=True,
            ),
            row=row, col=col
        )
        return fig
