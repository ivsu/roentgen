import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go


def create_dataset(num_series, num_steps, period=24, mu=1, sigma=0.1):
    # корректируем до большего кратного period
    estimated_steps = num_steps + (period - num_steps % period)
    # create target: noise + pattern
    # noise
    noise = np.random.normal(mu, sigma, size=(num_series, estimated_steps))

    # pattern - sinusoid with different phase
    k = 1 / period - 1
    sin_minus_pi_to_pi = np.sin(
        np.tile(np.linspace(-np.pi, 2 * np.pi * k - np.pi, period), int(estimated_steps / period))
    )
    sin_zero_to_2pi = np.sin(
        np.tile(np.linspace(0, 2 * np.pi * k, period), int(estimated_steps / period))
    )
    pattern = np.concatenate((
        np.tile(sin_minus_pi_to_pi.reshape(1, -1), (int(np.ceil(num_series / 2)), 1)),
        np.tile(sin_zero_to_2pi.reshape(1, -1), (int(np.floor(num_series / 2)), 1)),
    ), axis=0)

    target = noise + pattern

    # create time features: use target one period earlier, append with zeros
    feat_dynamic_real = np.concatenate(
        (np.zeros((num_series, period)), target[:, :-period]), axis=1
    )

    # create categorical static feats: use the sinusoid type as a categorical feature
    feat_static_cat = np.concatenate(
        (
            np.zeros(int(np.ceil(num_series / 2))),
            np.ones(int(np.floor(num_series / 2))),
        ),
        axis=0,
    )

    return target[:, :num_steps], feat_dynamic_real[:, :num_steps], feat_static_cat


def generate_debug_df(channel_names, dots_per_period):

    n_channels = len(channel_names)

    start_period = pd.Period("2022-01-03", freq="W-SUN")

    # define the parameters of the dataset
    custom_ds_metadata = {
        "num_series": n_channels,
        "num_steps": 121,
        "prediction_length": None,
        "freq": "W-SUN",
        "start": [start_period for _ in range(n_channels)],
    }

    data_out = create_dataset(
        custom_ds_metadata["num_series"],
        custom_ds_metadata["num_steps"],
        dots_per_period,
    )

    target, feat_dynamic_real, feat_static_cat = data_out

    # индекс - период
    rng = pd.period_range(start_period,
                          start_period + (custom_ds_metadata["num_steps"] - 1) * pd.offsets.Week(weekday=6),
                          freq='W-SUN')

    return pd.DataFrame(target.T * 1000, columns=channel_names, index=rng)


def show_data(df, channel_names):
    fig = go.Figure()

    data_index = 0

    def get_data(index):
        return df[channel_names[index]].copy()

    target = get_data(data_index)

    fig.add_trace(go.Scatter(
        x=df.index.to_timestamp(),
        y=target,
        mode='lines+markers',
        name=channel_names[data_index],
    ))

    print(channel_names)
    filter_buttons = []
    for i, channel in enumerate(channel_names):
        target = get_data(i)
        print(target.iloc[3])
        filter_buttons.append(dict(
            args=[dict(
                y=[target],
                x=[df.index.to_timestamp()],
                ),
                dict(title=f'Тестовые данные по модальности: {channel}'),
            ],
            label=channel,
            method='update'
        ))

    fig.update_xaxes(title_text="Недели года")
    fig.update_yaxes(title_text="Количество исследований")
    fig.update_layout(
        title_text=f'Тестовые данные по модальности (init): {channel_names[data_index]}',
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


if __name__ == '__main__':
    os.chdir('..')
    # logger.setup(level=logger.INFO, layout='debug')

    from workplan.datamanager import COLLAPSED_CHANNELS, ALL_CHANNELS

    channel_names = ALL_CHANNELS

    df = generate_debug_df(channel_names, dots_per_period=13)

    print(df.tail(18))

    show_data(df, channel_names)

