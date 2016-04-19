from dataGenerator import *
import kernelNaiveBayes as knb
import matplotlib.pyplot as plt

# Make HMM
N = 3 #10
numObs = N+1
numState = 2 #5
l = 400
min_sigma = 0.5 #0.1
T = makeTransitionMatrix(numState, min_sigma)
O = makeObservationMatrix(numState, numObs, min_sigma)
initDist = makeDistribution(numState)
Data = generateData_general(T, O, initDist, l)

print T
print O

#Learn HMM
xRange = np.matrix(np.linspace(Data.min(),Data.max(),numObs)).T 

noise = np.random.rand(Data.shape[0],Data.shape[1])/1e3 #jiggle the data points a little bit to improve conditioning
O_hat = knb.kernXMM(Data+noise,2,xRange,var=0.5)

#Plot results for comparison
fig = plt.figure(1)
ax = fig.add_subplot(1,2,1)
ax.plot(xRange,O_hat[:,0])
ax.set_title('pdf of component h=0'); ax.set_xlabel("x");ax.set_ylabel("est pdf"); 
ax = fig.add_subplot(1,2,2);
ax.plot(xRange,O_hat[:,1])
ax.set_title('pdf of component h=1'); ax.set_xlabel("x");ax.set_ylabel("est pdf"); 

fig = plt.figure(2)
ax = fig.add_subplot(1,2,1)
ax.plot(xRange,O[:,0])
ax.set_title('pdf of component h=0'); ax.set_xlabel("x");ax.set_ylabel("true pdf"); 
ax = fig.add_subplot(1,2,2);
ax.plot(xRange,O[:,1])
ax.set_title('pdf of component h=1'); ax.set_xlabel("x");ax.set_ylabel("true pdf"); 
plt.show()
