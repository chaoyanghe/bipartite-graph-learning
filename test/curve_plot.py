import matplotlib.pyplot as plt
from pylab import *
import numpy as np

# data to plot
x_pos = np.arange(0, 300)
y1_value = 0.3 * np.arange(0, 300)
y2_value = 0.4 * np.arange(0, 300)

# figure
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x_pos, y1_value, '--g', label='Adapt')
ax.plot(x_pos, y2_value, 'r', label='Full')
plt.xlabel('x_pos', fontsize=20)
plt.ylabel('performance', fontsize=20)
plt.xticks(fontsize=16)
tick_params(which='major', direction='in')
tick_params(top=False, bottom=True, left=True, right=True)
plt.yticks(fontsize=16)
# plt.legend('y1_value','y2_value')
plt.grid
plt.axis([0, 400, 0, 450])
plt.title('Testing Accuracies', fontsize=20)
ax.legend(loc='upper center', fontsize=20)
plt.show()