import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.font_manager as font_manager
from matplotlib import colormaps


def get_colors_from_cmap(cmap_name, num_colors):
    cmap = colormaps[cmap_name]
    return [cmap(i / (num_colors - 1)) for i in range(num_colors)]


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


if __name__ == '__main__':
    Ego_type = ["DQN-Ego", "IDM-Ego"]
    Lanes_type = ["2lanes", "3lanes"]
    font_props = font_manager.FontProperties(family='Times New Roman', size=12)
    colors = get_colors_from_cmap('viridis', len(Ego_type)*len(Lanes_type))
    count = 0

    plt.figure()
    for ego in Ego_type:
        for lane in Lanes_type:

            rewards = pd.read_csv(
                f"AdvLogs/{ego}-{lane}/run-{ego}-{lane}-tag-Mean Reward per step.csv")

            plt.plot(rewards['Step'], smooth(rewards['Value'], sm=50), color=colors[count], label=f"{ego}-{lane}")
            count += 1
            # 不使用平滑处理
            # ax1.plot(len_mean['Step'], len_mean['Value'], color="red",label='all_data')

    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Mean Reward")
    plt.savefig("AdvLogs/Learning_Curve.png", dpi=400)
    plt.show()
