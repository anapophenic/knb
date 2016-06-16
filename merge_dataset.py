import scipy.io
import numpy as np

if __name__ == '__main__':

    filename1 = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_E1_chr1_binsize100.mat'
    filename2 = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_E2_chr1_binsize100.mat'
    filename = 'Data_Intact/cndd/emukamel/HMM/Data/Binned/allc_AM_E_chr1_binsize100.mat'
    
    mat1 = scipy.io.loadmat(filename1)  
    coverage1 = mat1['h']
    methylated1 = mat1['mc']
    mat2 = scipy.io.loadmat(filename2)  
    coverage2 = mat2['h']
    methylated2 = mat2['mc']
    
    print np.shape(coverage1)
    print np.shape(coverage2)
    
    coverage = coverage1 + coverage2
    methylated = methylated1 + methylated2
    
    mat = {}
    mat['h'] = coverage
    mat['mc'] = methylated
    
    scipy.io.savemat(filename, mat);
