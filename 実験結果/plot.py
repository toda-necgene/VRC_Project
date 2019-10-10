"""
1次元散布図を並列に表示するスクリプト
"""
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    d = np.loadtxt("log.csv", delimiter=",")
    d = d.T
    name = ["resnet9*256", "resnet9*128", "resnet6*128", "depthwise6*128", "depthwise9*256"]
    for i in range(d.shape[0]):
        t = d[i, 0]
        data = d[i, 1:]
        n = np.ones_like(data) * (i+1)
        plt.scatter(n, data, label=name[i])
    plt.legend()
    plt.show()
    for i in range(d.shape[0]):
        plt.subplot(2, d.shape[0], i+1)
        t = d[i, 0]
        data = d[i, 1:]
        plt.title(name[i])
        plt.hist(data, bins=10)
        plt.subplot(2, d.shape[0], i+1+d.shape[0])
        plt.ylim(0.04, 0.4)
        plt.boxplot(data)
        table = plt.table(cellText=[["minimum", "mean", "stddev"], ["%f"% float(np.min(data)), "%f" % float(np.mean(data)), "%f" % float(np.std(data))]])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
    plt.show()
