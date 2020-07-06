import numpy as np
import pickle
import math


def crete_channel(h1,h2,variance):
    ch = {'h1': h1, 'h2': h2, 'var': variance}
    return ch


def channel_output(ch,ik,ik_1):
    nk = np.random.normal(0,ch['var'])
    xk = ch['h1']*ik + ch['h2']*ik_1 + nk
    return xk


def get_cluster_means(ch,filename):
    f = open(filename)
    states = {0: {}, 1: {}, 2: {}, 3: {}}

    for i in range(4):
        for j in range(2):
            states[i][j] = {}
            states[i][j]['mean'] = [0, 0]
            states[i][j]['count'] = 0

    states[0][0]['next'] = states[0]
    states[0][1]['next'] = states[2]
    states[1][0]['next'] = states[0]
    states[1][1]['next'] = states[2]
    states[2][0]['next'] = states[1]
    states[2][1]['next'] = states[3]
    states[3][0]['next'] = states[1]
    states[3][1]['next'] = states[3]

    ik_1 = 0
    ik_2 = 0
    xk_1 = channel_output(ch,ik_1,ik_2)

    while True:
        bit = f.read(1)
        if not bit:
            f.close()
            break
        ik = int(bit)
        xk = channel_output(ch,ik,ik_1)
        u = [xk, xk_1]
        xk_1 = xk
        s = 2*ik_1 + ik_2
        ik_2 = ik_1
        ik_1 = ik
        states[s][ik]['mean'] = np.add(u, states[s][ik]['mean'])
        states[s][ik]['count'] += 1

    total = 0
    for i in range(4):
        for j in range(2):
            if not states[i][j]['count'] == 0:
                states[i][j]['mean'] = np.divide(states[i][j]['mean'], states[i][j]['count'])
            total += states[i][j]['count']
    #print total
    return states


def get_distorted_data(ch, filename):
    f = open(filename)
    data = []
    ik_1 = 0
    xk_2 = channel_output(ch, ik_1, ik_1)
    xk_1 = channel_output(ch, ik_1, ik_1)
    x = [xk_1,xk_2]
    data.append(x)

    while True:
        bit = f.read(1)
        if not bit:
            f.close()
            break
        ik = int(bit)
        xk = channel_output(ch, ik, ik_1)
        u = [xk, xk_1]
        xk_1 = xk
        ik_1 = ik
        data.append(u)

    return data


def viterbi(states,data):
    trellis = []
    samples = len(data)
    s1 = 0
    s2 = 0
    t = 0
    for i in range(samples):
        trellis.append([])
        d = data[i]
        for j in range(4):
            trellis[i].append({})
            trellis[i][j]['prev'] = None
            trellis[i][j]['distance'] = 0
            trellis[i][j]['level'] = i
            if i > 0:
                if j == 0:
                    s1 = 0
                    s2 = 1
                    t = 0
                elif j == 1:
                    s1 = 2
                    s2 = 3
                    t = 0
                elif j == 2:
                    s1 = 0
                    s2 = 1
                    t = 1
                elif j == 3:
                    s1 = 2
                    s2 = 3
                    t = 1
                trellis[i][j]['transition'] = str(t)
                u1 = states[s1][t]['mean']
                u2 = states[s2][t]['mean']
                d1 = math.sqrt(math.pow(d[0] - u1[0], 2) + math.pow(d[1] - u1[1], 2))
                d2 = math.sqrt(math.pow(d[0] - u2[0], 2) + math.pow(d[1] - u2[1], 2))
                d1 = d1 + trellis[i - 1][s1]['distance']
                d2 = d2 + trellis[i - 1][s2]['distance']

                if d1 < d2:
                    trellis[i][j]['distance'] = d1
                    trellis[i][j]['prev'] = trellis[i - 1][s1]
                else:
                    trellis[i][j]['distance'] = d2
                    trellis[i][j]['prev'] = trellis[i - 1][s2]
    min_dist = None
    best_node = None
    for j in range(4):
        if min_dist is None or min_dist > trellis[samples-1][j]['distance']:
            min_dist = trellis[samples-1][j]['distance']
            best_node = trellis[samples-1][j]
    recovered = ''
    while not best_node['prev'] is None:
        recovered = best_node['transition'] + recovered
        best_node = best_node['prev']
    return recovered


def calculate_accuracy(filename, recovered):
    f = open(filename)
    correct = 0
    bits = len(recovered)
    for i in range(bits):
        char = f.read(1)
        if not recovered[i] == char:
            print 'Bit flipped at position', (i+1)
        else:
            correct += 1
    accuracy = (correct*100.0)/bits
    return accuracy


np.random.seed(1)
h_1 = 0.5
h_2 = 1
noise_variance = 0.25
train_file = 'train.txt'
test_file = 'test.txt'
recovery_file = 'recovered.txt'
states_info_file = 'states.pkl'

channel = crete_channel(h_1,h_2,noise_variance)

#states_info = get_cluster_means(channel,train_file)
#output_ = open(states_info_file, 'wb')
#pickle.dump(states_info, output_, pickle.HIGHEST_PROTOCOL)
#output_.close()

input_ = open(states_info_file, 'rb')
states_info = pickle.load(input_)
input_.close()

distorted_data = get_distorted_data(channel, test_file)
recovered_data = viterbi(states_info, distorted_data)

recovered_file = open(recovery_file, 'w')
recovered_file.write(recovered_data)
recovered_file.close()
accuracy = calculate_accuracy(test_file, recovered_data)
print 'Accuracy of recovery:',accuracy
