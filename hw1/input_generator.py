import numpy as np
import matplotlib.pyplot as plt
import random

# ground truth function
x = np.linspace(-1, 1, 50)
# y = 2*x + 1
ground_truth = x**3 + x**2 + 3

# sample points
x_list = np.linspace(-1, 1, 20)
y_list = [x**3 + x**2 + 3 for x in x_list]

# print('x_list size = %d' % len(x_list))
# print('y_list size = %d' % len(y_list))

# print(x_list)
# print(y_list)

# write into a file
f = open('input_data.txt', 'w+')
for i in range(len(x_list)):
    f.write('%f, %f\n' % (x_list[i], y_list[i]))
f.close()

plt.plot(x, ground_truth)
plt.scatter(x_list, y_list, c='green', edgecolors='none')
plt.show()
