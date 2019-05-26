import matplotlib.pyplot as plt
import numpy as np
from pylab import *

# data to plot
n_groups = 2
Node2Vec = (15, 10)
VGAE = (0, 0 )
GraphSAGE = (150, 120)
ABCGraph_MLP = (13.5, 12)
ABCGraph_Adv = (11, 12)


# create plot
fig, ax = plt.subplots(figsize=(8, 4))
index = 3.5*np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

x_pos = (0, 1.5*bar_width, 2*1.5*bar_width, 3*1.5*bar_width, 4*1.5 *bar_width)


rects1 = plt.bar(index, Node2Vec, bar_width,
alpha=opacity,
color='c',
label='Node2Vec')


rects2 = plt.bar(index + 1 * 1.5 * bar_width, VGAE, bar_width,
alpha=opacity,
color='g',
label='GAE')

rects3 = plt.bar(index + 2 * 1.5 * bar_width, GraphSAGE, bar_width,
                 alpha=opacity,
                 color='b',
                 label='GraphSAGE')

rects4 = plt.bar(index + 3 * 1.5 * bar_width, ABCGraph_MLP, bar_width,
                 alpha=opacity,
                 color='m',
                 label='ABCGraph_MLP')
rects5 = plt.bar(index + 4 * 1.5 * bar_width, ABCGraph_Adv, bar_width,
                 alpha=opacity,
                 color='y',
                 label='ABCGraph_Adv')
# plt.xlabel('')
plt.ylabel('Training Time (min)', fontsize=16)
# plt.title('Training Time Comparison', fontsize=16)
plt.axis([-1, 6, 0, 140])
plt.xticks(index + 3 * bar_width, ('CPU','GPU'), fontsize=16)
plt.text(1.5 * bar_width, 5, 'N/A', ha='center',va='center')
plt.text(11.5 * bar_width, 5, 'N/A', ha='center',va='center')

tick_params(which='major', direction='in')
tick_params(top=False, bottom=True, left=True, right=True)
plt.yticks(fontsize=16)
plt.legend(fontsize=16, loc='upper left')

plt.tight_layout()
plt.show()
# fig.savefig("training_time_comparison.eps")
