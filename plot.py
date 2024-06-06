import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.font_manager as font_manager
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def smooth(data, sm=1):
    """
        smooth the input data
    """
    if sm > 1:
        z = np.ones_like(data)
        y = np.ones(sm) * 1.0
        smoothed = np.convolve(y, data, "same") / np.convolve(y, z, "same")
        return smoothed
    else:
        return data


def extract_event_values(event_file, tag):
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()
    if tag not in event_acc.Tags()['scalars']:
        return [], []

    events = event_acc.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    return steps, values


def collect_events_by_prefix(log_dir, tag='Mean Reward'):
    reward_datas = {}
    min_step_len = 1e6
    for root, dirs, files in os.walk(log_dir):
        for dir_name in dirs:
            parts = dir_name.split('-')
            prefix = '-'.join(parts[:-1])
            if prefix not in reward_datas:
                reward_datas[prefix] = []
            full_dir_path = os.path.join(root, dir_name)
            for file in os.listdir(full_dir_path):
                if "events" in file:
                    event_file = os.path.join(full_dir_path, file)
                    steps, values = extract_event_values(event_file, tag)
                    min_step_len = min(len(steps), min_step_len)
                    reward_datas[prefix].append((steps, values))

    return reward_datas, min_step_len


def plot_data(datas, min_step_len, save_dir, tag):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    for prefix, data_list in datas.items():
        combined_data = []
        sm = min_step_len // 20
        for steps, values in data_list:
            smooth_values = smooth(values, sm=sm)
            df = pd.DataFrame({'step': steps, 'smoothed_value': smooth_values})
            df = df.iloc[:min_step_len]
            combined_data.append(df)
        smoothed_df = pd.concat(combined_data)

        sns.lineplot(
            data=smoothed_df,
            x='step',
            y='smoothed_value',
            ax=ax,
            label=prefix,
            estimator='mean',
            errorbar=('ci', 90),
            err_kws={"alpha": 0.2, "linewidth": 0.1}
        )

    ax.set_title("Training Progress", fontname="Times New Roman", fontsize=16)
    ax.set_xlabel("Episode", fontname="Times New Roman", fontsize=16)
    ax.set_ylabel(str(tag), fontname="Times New Roman", fontsize=16)
    plt.legend(loc="lower right")
    path = os.path.join(save_dir, f'{tag}.png')
    plt.savefig(path, dpi=400)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default="AdvLogs")
    parser.add_argument('--tag', type=str, default="Mean Reward", choices=["Mean Reward", "Loss"])
    parser.add_argument('--save_dir', type=str, default="image")
    args = parser.parse_args()
    # get reward
    datas, min_step_len = collect_events_by_prefix(log_dir=args.log_dir, tag=args.tag)
    # plot reward
    plot_data(datas, min_step_len, args.save_dir, args.tag)



