"""
Jacob Maurer
8/3/2026
Prupose: Create nice plots for paper
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    data = pd.read_csv("./data_table.csv", delimiter='\t')
    plt.bar(data["Column 1"], )
    # data_std = pd.read_csv("./data_table_std.csv", delimiter='\t')
    # full_data = pd.merge(data, data_std, on="Column 1")
    # print(full_data.head())

    # group_a_means, group_a_std = full_data["A"], full_data["As"]
    # group_b_means, group_b_std = full_data["B"], full_data["Bs"]

    # ind = np.arange(len(group_a_means))  # the x locations for the groups
    # width = 0.35  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(ind - width/2, group_a_means, width, yerr=group_a_means,
    #                 label='Men')
    # rects2 = ax.bar(ind + width/2, group_b_means, width, yerr=group_b_means,
    #                 label='Women')

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_ylabel('Avg. Reward')
    # ax.set_title('Scores by group and gender')
    # ax.set_xticks(ind)
    # ax.set_xticklabels(data["Column 1"])
    # ax.legend()


    # def autolabel(rects, xpos='center'):
    #     """
    #     Attach a text label above each bar in *rects*, displaying its height.

    #     *xpos* indicates which side to place the text w.r.t. the center of
    #     the bar. It can be one of the following {'center', 'right', 'left'}.
    #     """

    #     ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    #     offset = {'center': 0, 'right': 1, 'left': -1}

    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(offset[xpos]*3, 3),  # use 3 points offset
    #                     textcoords="offset points",  # in both directions
    #                     ha=ha[xpos], va='bottom')


    # autolabel(rects1, "left")
    # autolabel(rects2, "right")

    # fig.tight_layout()
    # plt.show()