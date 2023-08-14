import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def tensorboard_smoothing(x, smooth=0.99):
    x = x.copy()
    weight = smooth
    for i in range(1, len(x)):
        x[i] = (x[i - 1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x


if __name__ == '__main__':
    Ego_type = ["DQN-Ego", "IDM-Ego"]
    Lanes_type = ["2lanes", "3lanes"]
    for ego in Ego_type:
        for lane in Lanes_type:
            fig, ax1 = plt.subplots(1, 1)  # a figure with a 1x1 grid of Axes
            # 设置上方和右方无框
            ax1.spines['top'].set_visible(False)  # 不显示图表框的上边框
            ax1.spines['right'].set_visible(False)
            losses_mean = pd.read_csv(
                f"./{ego}-{lane}/{ego}-{lane}-10000-losses.csv")
            rewards = pd.read_csv(
                f"./{ego}-{lane}/{ego}-{lane}-10000-losses.csv")

            reward_mean = np.mean(rewards['Value'][:-10])
            # 设置折线颜色，折线标签
            # 使用平滑处理
            ax1.plot(losses_mean['Step'], tensorboard_smoothing(losses_mean['Value'], smooth=0.6), color="red", label='losses')
            # 不使用平滑处理
            # ax1.plot(len_mean['Step'], len_mean['Value'], color="red",label='all_data')

            # s设置标签位置，lower upper left right，上下和左右组合
            plt.legend(loc='lower right')

            ax1.set_xlabel("frame")
            ax1.set_ylabel("losses")
            ax1.set_title(f"{ego}-{lane} reward: {reward_mean:.2f}")
            plt.show()
            # 保存图片，也可以是其他格式，如pdf
            fig.savefig(fname=f"{ego}-{lane}-10000-losses" + '.png', format='png')
