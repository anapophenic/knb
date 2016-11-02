import scipy.io
import numpy as np

if __name__ == '__main__':

    chrs = [str(a) for a in range(1,20,1)]
    chrs.append('X')
    chrs.append('Y')
    
    #cells = ['E1', 'E2', 'V8', 'V9', 'P13P14', 'P15P16']
    
    for ch in chrs:

        filename1 = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_E1_chr' + str(ch) + '_binsize100.mat'
        filename2 = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_E2_chr' + str(ch) + '_binsize100.mat'
        filename = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_E_chr' + str(ch) + '_binsize100.mat'
        
        mat1 = scipy.io.loadmat(filename1)  
        coverage1 = mat1['h']
        methylated1 = mat1['mc']
        mat2 = scipy.io.loadmat(filename2)  
        coverage2 = mat2['h']
        methylated2 = mat2['mc']
        
        print np.shape(coverage1)
        print np.shape(coverage2)
        
        l = min(np.shape(coverage1)[0], np.shape(coverage2)[0])
        print l
        
        coverage = coverage1[:l, :] + coverage2[:l, :]
        methylated = methylated1[:l, :] + methylated2[:l, :]
        
        mat = {}
        mat['h'] = coverage
        mat['mc'] = methylated
        
        scipy.io.savemat(filename, mat);
