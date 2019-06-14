import math
import numpy as np

if __name__ == "__main__":
    # get user input
    a = int(input('Parameter a for the initial beta prior: '))
    b = int(input('Parameter b for the initial beta prior: '))

    # read input
    file_name = 'part2_input.txt'
    with open(file_name) as f:
        content = f.readlines()
    trials = [x.strip('\n') for x in content]

    a_bs = np.zeros((len(trials) + 1, 2))
    a_bs[0] = [a, b]
    
    likelihoods = []
    for idx, data in enumerate(trials):
        m = data.count('1') # number of 1
        N = len(data)       # total number of tossing a coin
        p = m / N
        likelihood = (math.factorial(N) / (math.factorial(m) * math.factorial(N - m))) * (p ** m) * ((1 - p) ** (N - m))
        likelihoods.append(likelihood)

        a_bs[idx + 1] = [a_bs[idx, 0] + m, a_bs[idx, 1] + (N - m)]
    
    for idx, data in enumerate(trials):
        print('case %d: %s' % (idx + 1, data))
        print('Likelihood: %f' % (likelihoods[idx]))
        print('Beta prior:     a = %d b = %d' % (a_bs[idx, 0], a_bs[idx, 1]))
        print('Beta posterior: a = %d b = %d\n' % (a_bs[idx + 1, 0], a_bs[idx + 1, 1]))