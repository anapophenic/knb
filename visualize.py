import numpy as np
import matplotlib.pyplot as plt
import feature_map as fm
import binom_hmm as bh
import matplotlib.collections as collections

def print_v(p, vec_title):
    print_m(np.array([p.tolist()]), vec_title)

def print_m(M, mat_title):
    #M = np.random.random((10,10))
    fig = plt.figure(1)
    plt.matshow(M, interpolation='nearest', cmap=plt.cm.Spectral)
    fig.savefig(mat_title)
    plt.show(block=False)
    plt.close(fig)

def get_fig_title(path_name, ce, ch, l, s, m, n, phi, ctxt):
    return path_name + '/' + 'cell = ' + ce + \
           '_chr = ' + ch + '_l = ' + str(l) + \
           '_s = ' + str(s) + '_m = ' + str(m) + \
           '_n = ' + str(n) + '_phi = ' + fm.phi_name(phi) + \
           '_ctxt = ' + bh.ctxt_name(ctxt)


def print_feature_map(C_h, color_scheme, feature_map_title):
    n, m = np.shape(C_h)

    fig = plt.figure(1)
    plt.hold(True)
    ax = fig.add_subplot(1,1,1)
    ax.set_title(r'Expected Feature Map given Hidden States')

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\mathbb{E}[\phi(x,t)|h]$')

    #print color_scheme

    for i in range(m):
        print i
        print C_h[:,i]
        plt.plot(bh.unif_partition(n), C_h[:,i], color=color_scheme[i], linewidth=3)

    fig.savefig(feature_map_title)
    # save the figure to file
    plt.hold(False)
    plt.close(fig)
    #print 'Refining using Binomial Knowledge'

def get_color_scheme(h, m):
    h = h.tolist()
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    freq = [(i, h.count(i)) for i in xrange(m)]
    sorted_freq = sorted(freq, key=lambda x: x[1], reverse=True)
    color_scheme = {};
    for i in xrange(m):
        color_scheme[sorted_freq[i][0]] = colors[i]

    return color_scheme


def browse_states(h, h_name, color_scheme):
    #fig = plt.figure(1)
    l = len(h);
    m = max(color_scheme.keys()) + 1

    plt.hold(True)
    fig, ax = plt.subplots(1, 1)
    for i in range(m):
        #plt.bar(xrange(l), [h[j] == i for j in xrange(l)], 1, color=color_scheme[i], edgecolor='none')
        collection = collections.BrokenBarHCollection.span_where(range(l), ymin=0, ymax=1, where=[h[j] == i for j in xrange(l)], facecolor=color_scheme[i], alpha=0.5, edgecolor='none')
        ax.add_collection(collection)

    plt.axis([0, l, 0, 1])
    plt.hold(False)
    fig.savefig(h_name)
    #for ax, i in zip(axes, xrange(l)):
    plt.close(fig)
