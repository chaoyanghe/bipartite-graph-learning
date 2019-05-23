import matplotlib.pyplot as plt
import numpy as np
from pylab import *
# data to plot
n_groups = 2
Adapt = (90, 55)
Full = (85, 62)
IID = (40, 34)
Node = (32, 59)

# create plot
fig, ax = plt.subplots()
index = 3*np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, Adapt, bar_width,
alpha=opacity,
color='c',
label='Adapt')


rects2 = plt.bar(index + 1 * 1.5 * bar_width, Full, bar_width,
alpha=opacity,
color='g',
label='Full')

rects3 = plt.bar(index + 2 * 1.5 * bar_width, IID, bar_width,
                 alpha=opacity,
                 color='b',
                 label='IID')

rects3 = plt.bar(index + 3 * 1.5 * bar_width, Node, bar_width,
                 alpha=opacity,
                 color='m',
                 label='Node')

# plt.xlabel('')
plt.ylabel('Performance', fontsize=20)
plt.title('Testing Accuracies', fontsize=20)
plt.xticks(index + 0.5 * 3 * 1.5 * bar_width, ('PubMed', 'Reddit'), fontsize=16)

tick_params(which='major', direction='in')
tick_params(top=False, bottom=True, left=True, right=True)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)

plt.tight_layout()
plt.show()