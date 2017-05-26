import numpy as np
import matplotlib.pyplot as plt
import feature_map as fm
import binom_hmm as bh
import matplotlib.collections as collections
import os
import scipy.io as io
import matplotlib.pylab as plt
import matplotlib.patches as patches
import utils as ut

def directory_setup(mod):
    try:
        os.stat(mod.path_name)
    except:
        os.mkdir(mod.path_name)

    try:
        os.stat(mod.path_name + '/figs')
    except:
        os.mkdir(mod.path_name + '/figs')

'''
def print_v(p, vec_title, state_labels):
    print_m(np.array([p.tolist()]), vec_title, state_labels)

def print_m(M, mat_title, state_labels):
    fig = plt.figure(1)
    plt.matshow(M, interpolation='nearest', cmap=plt.cm.Spectral)

    fig.savefig(mat_title)
    plt.show(block=False)
    plt.close(fig)
'''
def show_m(mat, m_title, path_name, state_name, is_T):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 10
    fig_size[1] = 10
    plt.rcParams["figure.figsize"] = fig_size
    m = len(state_name)

    plt.hold(True)
    fig, ax = plt.subplots(1, 1)
    cax = plt.imshow(mat, interpolation='nearest', cmap=plt.cm.ocean, vmin=0, vmax=1)
    plt.xticks(range(m),  state_name)
    if is_T:
        plt.yticks(range(m), state_name)
    fig.colorbar(cax)
    fig.savefig(path_name + m_title)
    plt.close(fig)


def show_v(vec, v_title, path_name, p_ch):
    m = np.shape(vec)[0];
    mat = vec.reshape((1,m))
    show_m(mat, v_title, path_name, p_ch, False)

def group_name(ce_group):
    s = ''
    for ce in ce_group:
        s = s + ce

    return s

def get_fig_title(mod):

    return 'figs/' + 'ce_group = ' + group_name(mod.ce_group) + \
           '_chr = ' + mod.ch + '_l = ' + str(mod.l) + \
           '_s = ' + str(mod.s) + '_m_h = ' + str(mod.m_h) + \
           '_n = ' + str(mod.n) + '_phi = ' + fm.phi_name(mod.phi) + \
           '_ctxt_group = ' + bh.ctxt_name(mod.ctxt_group) + \
           '_' + mod.td_alg + '_' + mod.pp_alg

def get_sec_title(mod):

   return 'ce_group = ' + group_name(mod.ce_group) + \
          ', chr = ' + mod.ch + ', l = ' + str(mod.l) + \
          ', s = ' + str(mod.s) + \
          ', n = ' + str(mod.n) + ', phi = ' + fm.phi_name(mod.phi) + \
          ', ctxt_group = ' + bh.ctxt_name(mod.ctxt_group)

def print_hist(coverage, p_c):
    l = np.shape(coverage)
    N = np.amax(coverage)

    p_c[0,0] = 0
    p_c[0,:] = ut.normalize_v(p_c[0,:])

    mu = np.sum(coverage)/l
    poi_c = ut.truncated_poisson_pmf(mu, int(N))


    fig = plt.figure()
    plt.hold(True)

    plt.plot(p_c[0,:]*l,'b')
    plt.plot(poi_c*l, 'r')

    n, bins, patches = plt.hist(coverage, N, facecolor='green', alpha=0.75)
    fig.savefig('c_distn/c_distn' + '.png')
    plt.hold(False)
    plt.close(fig)

def print_feature_map(mod):
    n, m = np.shape(mod.C_h)
    r = len(mod.lims) - 1

    fig = plt.figure(1)
    plt.hold(True)

    fig.set_size_inches(40, 10)
    for j in range(r):
        ax = fig.add_subplot(1,r,j+1)
        ax.set_title(r'Expected Feature Map given Hidden States')

        ax.set_xlabel(r'$t$')
        ax.set_ylabel(r'$\mathbb{E}[\phi(x,t)|h]$')
        for i in range(m):
            #print i
            #print C_h[lims[j]:lims[j+1],i]
            #print heme[i]
            plt.plot(ut.unif_partition(mod.lims[j+1]-mod.lims[j]), mod.C_h[mod.lims[j]:mod.lims[j+1],i], color=mod.color_scheme[i], linewidth=3)

    fig.savefig(mod.path_name + mod.feature_map_title)
    # save the figure to file
    plt.hold(False)
    plt.close(fig)

def get_color_scheme(h, m):
    h = h.tolist()
    #colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    colors = [tuple(np.random.rand(3)) for i in range(m)]
    freq = [(i, h.count(i)) for i in xrange(m)]
    sorted_freq = sorted(freq, key=lambda x: x[1], reverse=True)
    color_scheme = {};
    for i in xrange(m):
        color_scheme[sorted_freq[i][0]] = colors[i]

    return color_scheme

def default_color_scheme(m):
    colors = [tuple(np.random.rand(3)) for i in range(m)]
    return {i:colors[i] for i in range(m)}


def browse_states(h, path_name, posterior_title, color_scheme):
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
    fig.savefig(path_name + posterior_title)
    #for ax, i in zip(axes, xrange(l)):
    plt.close(fig)


# look at decoding - multiply by s on original seq

def print_bed(h, mod):
    l = len(h);

    bed_list = []
    for i in range(mod.m_h):
        bed_list.append([])

    for i in range(l):
        if (i == 0):
            i_start = 0
        elif (i == l-1 or h[i] != h[i-1]):
            bed_list[h[i-1]].append(("chr"+mod.ch, i_start*100*mod.s, i*100*mod.s))
            i_start = i

    for i in range(mod.m_h):
        f = open(mod.path_name + mod.bed_title + 'm = ' + str(mod.m_h) + 'i = ' + str(i) + '.bed', 'w')
        for ch, i_start, i_end in bed_list[i]:
            #base pair
            f.write(ch + '\t' + str(i_start) +'\t' + str(i_end) + '\n')
        f.close()

    return bed_list

def plot_bed(axarr, bed_list):
    m = len(bed_list)
    for i in range(m):
        for j in range(len(bed_list[i])):
            xstart = bed_list[i][j][1]
            xsize = bed_list[i][j][2] - bed_list[i][j][1]
            #print xstart, xsize
            #axarr[i].broken_barh([(xstart, xsize)], (0, 1), edgecolor=None)
            axarr[i].add_patch(patches.Rectangle((xstart, 0), xsize, 1))

def plot_meth(axarr, coverage, methylated):
    n_cells, l = np.shape(coverage)
    for i in range(n_cells):
        meth_rate = methylated[i,:].astype(float) / coverage[i,:]
        axarr[i].fill_between(range(0,l*100,100), meth_rate)
        axarr[i].set_xlim([0, l*100])

def plot_meth_full(axarr, coverage, methylated, s):
    n_cells, l = np.shape(coverage)
    for i in range(n_cells):
        meth_rate = methylated[i,:].astype(float) / coverage[i,:]
        axarr[3*i].fill_between(range(0,l*100*s,100), coverage[i,:])
        axarr[3*i].set_ylabel('coverage')
        axarr[3*i+1].fill_between(range(0,l*100*s,100), methylated[i,:])
        axarr[3*i+1].set_ylabel('methylated')
        axarr[3*i+2].fill_between(range(0,l*100*s,100), meth_rate)
        axarr[3*i+2].set_xlim([0, l*100*s])


def plot_bed_only(bed_list):
    m = len(bed_list)
    f, axarr = plt.subplots(m, 1, sharex=True)
    plot_bed(axarr, bed_list)
    plt.show()

def plot_meth_only(coverage, methylated):
    n_cells = np.shape(coverage)[0]
    f, axarr = plt.subplots(n_cells, 1, sharex=True)
    plot_meth(axarr, coverage, methylated)
    plt.show()

def plot_m_and_c(coverage, methylated):
    n_cells, l = np.shape(coverage)
    f, axarr = plt.subplots(3*n_cells, 1, sharex=True)

    for i in range(n_cells):
        for j in range(l):
            if coverage[i,j] == 0:
                coverage[i,j] = 1
        axarr[3*i].plot(range(0,l*100,100), coverage[i,:])
        axarr[3*i+1].plot(range(0,l*100,100), methylated[i,:])
        axarr[3*i+2].fill_between(range(0,l*100,100), methylated[i,:].astype(float) / coverage[i,:])
    plt.show()

def plot_meth_and_bed(coverage, methylated, bed_list, mod):
    m = len(bed_list)
    n_cells = np.shape(coverage)[0]
    plt.figure()
    # Get current size

    fig_size_temp = plt.rcParams["figure.figsize"]
    fig_size = fig_size_temp
    fig_size[0] = 500
    fig_size[1] = 40
    plt.rcParams["figure.figsize"] = fig_size

    fig, axarr = plt.subplots(n_cells*3+m, 1, sharex=True)
    plt.hold(True)
    plot_meth_full(axarr[:n_cells*3], coverage, methylated, mod.s)
    plot_bed(axarr[n_cells*3:n_cells*3+m], bed_list)
    for i in range(m):
        axarr[n_cells*3+i].set_ylabel(mod.state_name_h[i])

    fig.savefig(mod.path_name + mod.bed_title + 'contrast_m = ' + str(m) + 'n_cells = ' + str(len(mod.ce_group))+'_l='+str(mod.l)+'_l_test='+str(mod.l_test) + '.pdf')
    plt.hold(False)

    plt.rcParams["figure.figsize"] = fig_size_temp
    plt.close(fig)

def state_name(p_ch):
    m = np.shape(p_ch)[1]
    return [truncated_str(p_ch[:,i]) for i in range(m)]

def truncated_str(s):
    return str(['%.3f' % i for i in s])


def plot_meth_and_twobeds(coverage, methylated, mod):
    l1 = len(mod.bed_list_gt)
    l2 = len(mod.bed_list_h)
    n_cells = np.shape(coverage)[0]
    plt.figure()
    # Get current size

    fig_size_temp = plt.rcParams["figure.figsize"]
    fig_size = fig_size_temp
    fig_size[0] = 500
    fig_size[1] = 40
    plt.rcParams["figure.figsize"] = fig_size

    fig, axarr = plt.subplots(n_cells+l1+l2+1, 1, sharex=True)
    plt.hold(True)
    plot_meth(axarr[:n_cells], coverage, methylated)

    for i in range(0, l1):
        axn = n_cells+i
        plot_bed([axarr[axn]], [mod.bed_list_gt[i]])
        axarr[axn].set_ylabel(mod.state_name_gt[i])

    for i in range(0, l2):
        axn = n_cells+l1+1+i
        plot_bed([axarr[axn]], [mod.bed_list_h[i]])
        axarr[axn].set_ylabel(mod.state_name_h[i])

    fig.savefig(mod.path_name + mod.bed_title + 'l1 = ' + str(l1) + 'l2 = ' + str(l2) + 'n_cells = ' + str(n_cells)+'_l='+str(mod.l)+'_l_test='+str(mod.l_test))
    plt.hold(False)
    plt.rcParams["figure.figsize"] = fig_size_temp
    plt.close(fig)

def print_doc_header(mod):
    f = open(mod.path_name+mod.tex_name, 'w')
    s = "\\documentclass{article}\n\\usepackage{epsfig}\n\\usepackage[export]{adjustbox}\n\\usepackage{caption}\n\\usepackage{subcaption}\n\\usepackage{fullpage}\n\\usepackage{commath}\n\\usepackage{amssymb}\n\\usepackage[space]{grffile}\n\\usepackage{float}\n\\begin{document}\n"
    f.write(s)
    f.close()

def print_expt_setting(mod):
    s1 = "\\begin{verbatim}\n"
    s2 = "\\end{verbatim}\n"
    f = open(mod.path_name+mod.tex_name, 'a')
    f.write(s1)
    f.write(mod.sec_title)
    f.write(s2)
    f.close()

def print_table_header(mod):
    f = open(mod.path_name+mod.tex_name, 'a')
    s = "\\begin{figure}[H]\n\\begin{tabular}{cc}\n"
    f.write(s)
    f.close()

def print_fig_and(mod):
    f = open(mod.path_name+mod.tex_name, 'a')
    s = "\\begin{subfigure}[t]{0.4\\textwidth}\n\\includegraphics[width=\\textwidth]{"+ mod.posterior_title + "}\n\\end{subfigure}&\n"
    f.write(s)
    f.close()

def print_fig_bs(mod):
    f = open(mod.path_name+mod.tex_name, 'a')
    s = "\\begin{subfigure}[t]{0.4\\textwidth}\n\\includegraphics[width=\\textwidth]{"+ mod.feature_map_title + "}\n\\end{subfigure}\\\\\n"
    f.write(s)
    f.close()

def print_table_aheader(mod):
    f = open(mod.path_name+mod.tex_name, 'a')
    s = "\\end{tabular}\n\\end{figure}\n"
    f.write(s)
    f.close()

def print_doc_aheader(mod):
    f = open(mod.path_name+mod.tex_name, 'a')
    s = "\\end{document}\n"
    f.write(s)
    f.close()
    os.chdir(mod.path_name)
    os.system("pdflatex "+mod.tex_name)
    os.system("cd ..")

def save_moments(P_21, P_31, P_23, P_13, P_123, ch, ce_group, s, ctxt_group, l, path_name):
    moments = {};
    moments['P21'] = P_21;
    moments['P31'] = P_31;
    moments['P23'] = P_23;
    moments['P13'] = P_13;
    moments['P123'] = P_123;
    mat_name = 'ch = ' + str(ch) + ' ce_group = ' + str(ce_group) + ' s = ' + str(s) + ' ctxt_group = ' + str(ctxt_group) + ' temp' + 'l = ' + str(l) + '.mat';
    io.savemat(path_name + mat_name, moments)

def load_moments(filename):
    moments = io.loadmat(filename)
    P_21 = moments['P21'];
    P_31 = moments['P31'];
    P_23 = moments['P23'];
    P_13 = moments['P13'];
    P_123 = moments['P123'];
    return P_21, P_31, P_23, P_13, P_123

if __name__ == '__main__':

    path_name = 'merge_ctxts/'
    fig_name = 'take.pdf'
    tex_name = 'result.tex'

    print_doc_header(path_name,tex_name);
    print_table_header(path_name,tex_name);
    print_fig_and(path_name,fig_name,tex_name);
    print_fig_bs(path_name,fig_name,tex_name);
    print_table_aheader(path_name,tex_name);
    print_doc_aheader(path_name,tex_name);
