import os
from datetime import datetime, timedelta
from plotly import graph_objects as go
from plotly.subplots import make_subplots

from settings import DB_VERSION
from schedule.dataloader import DataLoader, MODALITIES
from workplan.datamanager import CHANNEL_LEGEND
from common.logger import Logger

logger = Logger(__name__)


def month_period(dt: datetime):
    month_name = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь',
                  'Июль', 'Август', 'Сентябрь', 'Октябрь', 'Ноябрь', 'Декабрь']
    month_index = dt.month - 1
    return month_name[month_index] + dt.strftime(' %Y')


class ShowSchedule:

    def __init__(self, read_data=False):
        self.schedule_version = 'final'
        self.data_layer = 'day_modality'

        assert 'ROENTGEN.SCHEDULE_START_DATE' in os.environ, 'В переменных среды не задана дата начала расчёта графика.'
        schedule_month_start = datetime.fromisoformat(os.environ['ROENTGEN.SCHEDULE_START_DATE'])
        if schedule_month_start.day != 1:
            # получаем дату начала месяца для отображения графика
            schedule_month_start = (schedule_month_start.replace(day=1) + timedelta(days=32)).replace(day=1)
        self.schedule_month_start = schedule_month_start

        self.df = self.read_data() if read_data else None

    def read_data(self):
        """Запрашивает из БД расписание работы врачей и возвращает его в виде датафрейма,
        готового для отображения."""
        dataloader = DataLoader()
        df = dataloader.get_schedule(self.schedule_version, self.schedule_month_start, self.data_layer)
        min_dt = datetime(1900, 1, 1)
        if DB_VERSION == 'SQLite':
            df['work_time'] = df['time_volume'].apply(
                lambda t: (datetime.strptime(t, '%H:%M:%S') - min_dt).total_seconds() / 3600 if t else '-')
            df['month_day'] = df['day'].str[-2:]
        else:
            # df['work_time'] = df['time_volume'].total_seconds() / 3600
            raise NotImplementedError()
        df.loc[df['availability'] == -1, 'modality'] = 'absent'
        return df

    def get_formatted_schedule(self):
        assert self.df is not None, 'Необходимо загрузить данные.'
        df = self.df.pivot(index='name', columns='month_day', values='work_time')
        df.reset_index(inplace=True, names='ФИО')
        return df

    def get_schedule_mods(self):
        assert self.df is not None, 'Необходимо загрузить данные.'
        df = self.df.pivot(index='name', columns='month_day', values='modality')
        df.reset_index(inplace=True, names='ФИО')
        return df

    def plot(self):
        df = self.get_formatted_schedule()
        mods = self.get_schedule_mods()

        col_list = [df[s] for s in list(df.columns)]
        col_names = list(df.columns)
        col_width = [80] + [40 for _ in range(len(col_names) - 1)]

        mod_color = ['Tan', 'DarkTurquoise', 'DeepSkyBlue', 'GreenYellow', 'Khaki', 'LightPink', 'DimGrey']
        color_map = dict(zip(MODALITIES + ['absent'], mod_color))
        legend_labels = [n for m, n in CHANNEL_LEGEND.items() if m in MODALITIES] + ['Отсутствует']

        def map_color(x):
            if x in color_map.keys():
                return color_map[x]
            return 'GhostWhite'

        colors = mods.map(map_color)
        color_list = [colors[s] for s in list(colors.columns)]

        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.7, 0.3],
            specs=[[{'type': 'table'}, {}],
                   [{'type': 'table', 'colspan': 2}, None],
                   ],
            insets=[dict(cell=(1, 1), type='table')],
            vertical_spacing=0.02,
            row_heights=[0.08, 0.92],
            # row_width=[10, 50]
        )
        fig.add_trace(  # легенда
            go.Table(
                header=dict(
                    fill_color='white',
                    height=2,
                ),
                cells=dict(
                    values=['Модальности:'] + legend_labels,
                    fill_color=['white'] + mod_color,
                    align='center',
                    height=26,
                    ),
            ),
            row=1, col=1,
        )
        fig.add_trace(  # график
            go.Table(
                columnwidth=col_width,
                header=dict(values=list(df.columns),
                            # fill_color=['paleturquoise', 'lavender'],
                            align='center'),
                cells=dict(values=col_list,
                           fill_color=color_list,
                           align='right')
            ),
            row=2, col=1,
        )
        # fig.layout['template']['data']['table'][0]['header']['fill']['color'] = 'rgba(0,0,0,0)'
        fig.update_layout(
            title_text=f'Расписание работы врачей-рентгенологов на {month_period(self.schedule_month_start)}',
            margin=dict(t=30, b=10),
            height=600,
            # showlegend=True,
        )
        fig.show()


if __name__ == '__main__':
    logger.setup(level=logger.INFO, layout='debug')

    # установим дату начала графика
    os.environ['ROENTGEN.SCHEDULE_START_DATE'] = '2024-05-01'

    show = ShowSchedule(read_data=True)
    show.plot()
