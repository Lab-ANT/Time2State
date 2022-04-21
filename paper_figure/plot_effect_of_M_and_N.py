import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.default'] = 'regular'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

COLOR = ["blue", "cornflowerblue", "mediumturquoise", "goldenrod", "gold", "yellow", "blue", "cornflowerblue", "mediumturquoise", "goldenrod", "yellow"]
# COLOR = ["blue", "cornflowerblue", "mediumturquoise", "goldenrod", "yellow"]
# x_label = np.arange(1, 60+2, 10)
# y_label = np.arange(1, 10+2, 2)
x_label = [50, 40, 30, 20, 10, 1]#[1,10,20,30,40,50]
y_label = [1,2,4,6,8,10]

# x, y: position
x = list(range(len(x_label)))
y = list(range(len(y_label)))

x_tickets = [str(_x) for _x in x_label]
y_tickets = [str(_x) for _x in y_label]

acc = [[0.582,0.5886,0.5904,0.58,0.5735, 0.5756],
[0.6388, 0.6427, 0.6531, 0.6418, 0.6364, 0.4525],
[0.6528, 0.6586, 0.6472, 0.6526, 0.6411, 0.4551],
[0.6657, 0.6636, 0.6624, 0.6648, 0.6553, 0.4588],
[0.6612, 0.6535, 0.6674, 0.6480, 0.6525, 0.4691],
[0.6463, 0.6563, 0.6525, 0.6583, 0.6374, 0.4564]]

acc = np.array(acc).T

# 注意顺序问题，见 [9]
# 2022.3.27：这里正常用，要反的**不**是这里，而是后文的 `acc.ravel()` 那里
xx, yy = np.meshgrid(x, y)  # 2022.3.27：这里正常用，要反的**不**是这里
# yy, xx = np.meshgrid(x, y)  # 2022.3.27：这里**别**反

color_list = []
for i in range(len(y)):
    c = COLOR[i]
    color_list.append([c] * len(x))
color_list = np.asarray(color_list)
# print(color_list)
# 2022.3.27：注意这里 `acc` 在 `ravel()` 之前要转置（`.T`）一下，见 [9]
xx_flat, yy_flat, acc_flat, color_flat = \
    xx.ravel(), yy.ravel(), acc.T.ravel(), color_list.ravel()
# print(xx_flat)
# print(yy_flat)

# fig, ax = plt.subplots(projection="3d")
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.bar3d(xx_flat - 0.35, yy_flat - 0.35, 0, 0.7, 0.7, acc_flat,
    color=color_flat,  # 颜色
    edgecolor="black",  # 黑色描边
    shade=True)  # 加阴影

# 座标轴名
ax.set_xlabel(r"$M$")
ax.set_ylabel(r"$N$")
ax.set_zlabel("ARI")

# 座标轴范围
ax.set_zlim((0, 1.01))

# 座标轴刻度标签
# 似乎要 `set_*ticks` 先，再 `set_*ticklabels`
# has to call `set_*ticks` to mount `ticklabels` to corresponding `ticks` ?
ax.set_xticks(x)
ax.set_xticklabels(x_tickets)
ax.set_yticks(y)
ax.set_yticklabels(y_tickets)

# 保存
# plt.tight_layout()
# fig.savefig("bar3d.png", bbox_inches='tight', pad_inches=0)
# plt.close(fig)
plt.show()