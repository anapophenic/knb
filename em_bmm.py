import data_import as di
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import utils as ut
import scipy as sp

def posterior_group(cov, met, p, pi):
    c, k = np.shape(p)
    post = np.zeros(k)

    for i in range(k):
        post[i] = pi[i]
        for r in range(c):
            post[i] *= sp.stats.binom.pmf(met[r], cov[r], p[r,i])
    post = ut.normalize_v(post)
    return post

def em_bmm_group(coverage, methylated, p, pi, iters=10):
    r, l = np.shape(coverage)

    post = np.zeros((k,l))
    for it in range(iters):
        print 'Iteration '+str(it)
        #E step
        for j in range(l):
            post[:,j] = posterior_group(coverage[:,j], methylated[:,j], p, pi)

        #M step
        pi = np.sum(post, axis=1) / l
        for i in range(k):
            for c in range(r):
                m_sum = sum([post[i,j] * methylated[c,j] for j in range(l)])
                c_sum = sum([post[i,j] * coverage[c,j] for j in range(l)])
                p[c,i] = m_sum / float(c_sum)

    return p, pi


def bmm_pmf(p, pi, cov):
    r, k = np.shape(p)
    component_pmf = np.zeros((k, cov+1))
    pmf = np.zeros((r,cov+1))

    for c in range(r):
        for i in range(k):
            for met in range(cov+1):
                component_pmf[i,met] = sp.stats.binom.pmf(met, cov, p[c,i]) * pi[i]
        pmf[c,:] = np.sum(component_pmf, axis=0)

    return pmf

if __name__ == '__main__':
    ctxt_group = [range(12,16)]
    ce_group = ['E','V']
    s = 1
    ch = '1'

    coverage, methylated, N, x_zipped, a = di.data_prep_ctxt_ce(ctxt_group, ce_group, s, ch);

    #plt.hist2d(coverage[0,:], methylated[0,:], bins=200, norm=LogNorm())
    l = np.shape(coverage)[1]
    idx = np.random.randint(l, size=1000)
    for k in range(2, 10):
        p, pi = em_bmm_group(coverage[:,idx], methylated[:,idx], k, 30)
        print p, pi
        #p, pi = em_bmm(coverage[0,idx], methylated[0,idx], k, 30)
        #print p, pi

        '''
        for c in range(200):
            c1 = coverage[0,:]
            m1 = methylated[0,:]
            idx = (c1 == c)

            fig = plt.figure()
            plt.hold(True)
            n, bins, patches = plt.hist(m1[idx], 50, normed=1, facecolor='green', alpha=0.75)
            pmf = bmm_pmf(p, pi, c)
            plt.plot(pmf, 'r')

            plt.title('c = '+ str(c))
            #plt.show(block=False)
            fig.savefig('hist_m_c/m_given_c_c = '+str(c) + '_k = ' + str(k) + '.png')
            plt.hold(False)
            plt.close(fig)
        '''

        '''
        def posterior(c, m, p, pi):
            k = np.shape(p)[0]
            post = np.zeros(k)
            for i in range(k):
                post[i] = sp.stats.binom.pmf(m, c, p[i])

            post = ut.normalize_v(post*pi)
            return post

        def em_bmm(coverage, methylated, k, iters=10):
            #initialize parameters
            p = np.random.rand(k)
            pi = ut.normalize_v(np.random.rand(k))

            l = np.shape(coverage)[0]
            post = np.zeros((k, l))

            for it in range(iters):
                #print 'Iteration '+str(it)
                #E step
                for j in range(l):
                    post[:,j] = posterior(coverage[j], methylated[j], p, pi)

                #M step
                pi = np.sum(post, axis=1) / l
                for i in range(k):
                    m_sum = sum([post[i,j] * methylated[j] for j in range(l)])
                    c_sum = sum([post[i,j] * coverage[j] for j in range(l)])
                    p[i] = m_sum / float(c_sum)

            return p, pi
        '''
