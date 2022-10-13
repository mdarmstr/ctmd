# Factorisation
# %%
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt 

# Only applicable for two matrices; khatri rao product
def krb(A,B):
    C = np.einsum('ij,kj -> ikj', A, B).reshape(A.shape[0] * B.shape[0], A.shape[1])
    return C

## Generate synthetic data
class Gaussian:
    def __init__(self,mn,std,amp,n):
        self.mean = mn
        self.stdev = std
        self.amplitude = amp
        self.n = n
        
    def gaussArray(self):
        x = np.arange(0, self.n)
        fx = self.amplitude * np.exp(-0.5*(x-self.mean)**2/self.stdev)
        return x, fx
 
#Create synthetic data with one factor for now   
class synData:
    def __init__(self,aN,bN,cN,dN):
        x, self.a = Gaussian(15,3,20,aN).gaussArray()
        
        #Is there any way of making this more Pythonic
        bTemp = np.random.rand(bN)
        self.b = bTemp / np.linalg.norm(bTemp)
        
        cTemp = np.arange(0,cN)
        self.c = cTemp / np.linalg.norm(cTemp)
        
        dTemp = np.random.rand(dN)
        self.d = dTemp / np.linalg.norm(dTemp)
        
    def tMat(self):
        # Okay this is where the magic is supposed to happen
        xTnsr = np.einsum('i,j,k->ijk', self.a, self.b, self.c)
        yMtrx = np.einsum('i,j->ij', self.a, self.d)
        return xTnsr, yMtrx

## Decomposition
## I am wondering if we can use inheritance to avoid the use of the if operators
class initialize: #Generate an object with random initialisations.
    def __init__(self, tnsr, mtrx, noComponents, Ainit = None, Binit = None, Cinit = None, Dinit = None):
        if Ainit == None:
            self.__A = np.random.random_sample([tnsr.shape[0], noComponents])
        else:
            self.__A = Ainit
        
        if Binit == None:
            self.__B = np.random.random_sample([tnsr.shape[1], noComponents])
        else:
            self.__B = Binit
        
        if Cinit == None:
            self.__C = np.random.random_sample([tnsr.shape[2], noComponents])
        else:
            self.__C = Cinit
        
        if Dinit == None:    
            self.__D = np.random.random_sample([mtrx.shape[1], noComponents]) #ok
        else:
            self.__D = Dinit
            
    def getA(self):
        return self.__A
    
    def getB(self):
        return self.__B
    
    def getC(self):
        return self.__C
    
    def getD(self):
        return self.__D

class CTMD:
    def __init__(self,tnsr,mtrx,noComponents, absThres=1e-12, relThres=1e-12, maxIter=1000, 
                 initial=None, isNormal = np.array([0,1,1,1],dtype=bool), isNonNeg = np.array([1,0,0,0],dtype=bool)):
        
        self.__tnsr = tnsr # Here we are going to try and make the object attributes private, so you can't modify them withou retraining the model.
        self.__mtrx = mtrx
        self.__noComponents = noComponents
        self.__absThres = absThres
        self.__relThres = relThres
        self.__maxIter = maxIter
        self.__isNormal = isNormal
        self.__isNonNeg = isNonNeg
        
        if initial == None: #what to do if this argument is empty
            initial = initialize(self.__tnsr,self.__mtrx,self.__noComponents)
            
        self.__A = initial.getA()
        self.__B = initial.getB()
        self.__C = initial.getC()
        self.__D = initial.getD()
        
        #Unfolding the tensor in each mode, so we only have to do it once
        self.__tnsr0 = tl.unfold(self.__tnsr,0)
        self.__tnsr1 = tl.unfold(self.__tnsr,1)
        self.__tnsr2 = tl.unfold(self.__tnsr,2)
                         
    # To do: add conditions for whether or not to normalise, or make the regression non-negative at each step
    def optA(self):
        self.__tnsr0Z = krb(self.__B,self.__C)
        
        self.__A = self.__tnsr0 @ self.__tnsr0Z @ np.linalg.pinv(self.__tnsr0Z.T @ self.__tnsr0Z)
        + self.__mtrx @ self.__D @ np.linalg.pinv(self.__D.T @ self.__D)
        
        if self.__isNormal[0] == True:
            self.__A /= np.linalg.norm(self.__A, axis=0)
            
    def optB(self):
        self.__tnsr1Z = krb(self.__A,self.__C)
                
        self.__B = self.__tnsr1 @ self.__tnsr1Z @ np.linalg.pinv(self.__tnsr1Z.T @ self.__tnsr1Z)
        
        if self.__isNormal[1] == True:
            self.__B /= np.linalg.norm(self.__B,axis=0)
        
    def optC(self):
        self.__tnsr2Z = krb(self.__B,self.__A)
        
        self.__C = self.__tnsr2 @ self.__tnsr2Z @ np.linalg.pinv(self.__tnsr2Z.T @ self.__tnsr2Z)
        
        if self.__isNormal[2] == True:
            self.__C /= np.linalg.norm(self.__C, axis=0)
    
    def optD(self):
        self.__D = self.__mtrx.T @ self.__A @ np.linalg.pinv(self.__A.T @ self.__A)
        
        if self.__isNormal[3] == True:
            self.__D /= np.linalg.norm(self.__D, axis=0)
                
    def fit(self):
        absThres = self.__absThres
        relThres = self.__relThres
        ## Here I am still working.
        ssr1 = 0.5 * np.linalg.norm(self.__tnsr - np.einsum('ij,kj,lj->ikl', self.__A, self.__B, self.__C)) ** 2 / np.linalg.norm(self.__tnsr) ** 2 + 0.5 * np.linalg.norm(self.__mtrx - np.einsum('ij,kj->ik', self.__A, self.__D)) ** 2 / np.linalg.norm(self.__mtrx) 
        ssr2 = 1
        
        while abs(ssr1 - ssr2) / ssr2 > relThres and abs(ssr1 - ssr2) > absThres: #ALS routine
            ssr1 = ssr2
            
            # Optimisations
            self.optA()
            self.optB()
            self.optC()
            self.optD()
            
            # Measure the error
            ssr2 = 0.5 * np.linalg.norm(self.__tnsr - np.einsum('ij,kj,lj->ikl', self.__A, self.__B, self.__C)) ** 2 / np.linalg.norm(self.__tnsr) ** 2 + 0.5 * np.linalg.norm(self.__mtrx - np.einsum('ij,kj->ik', self.__A, self.__D)) ** 2 / np.linalg.norm(self.__mtrx) 
            print(ssr2)
            
    def getA(self):
        return self.__A
    
    def getB(self):
        return self.__B
    
    def getC(self):
        return self.__C
    
    def getD(self):
        return self.__D

# %%   
test = synData(30,10,3,2)    
xTnsr, yMtrx = test.tMat()
    
mdl = CTMD(xTnsr, yMtrx, 1)
mdl.fit()

# predTnsr = np.einsum('i,j,k->ijk',)
predTnsr = np.einsum('ij,kj,lj->ikl', mdl.getA(), mdl.getB(), mdl.getC())
predMtrx = np.einsum('ij,kj->ik', mdl.getA(), mdl.getD())

fig1, axs1 = plt.subplots(2,2)
fig2, axs2 = plt.subplots(2,2)

# low microclimate
axs1[0,0].plot(predTnsr[:,:,0])
axs1[0,0].set_title("MicroSensors, Low")
axs1[0,0].set_ylabel("Temperature")
axs1[0,0].set_xlabel("Days")

axs1[0,1].plot(predTnsr[:,:,1])
axs1[0,1].set_title("MicroSensors, Med")
axs1[0,1].set_ylabel("Temperature")
axs1[0,1].set_xlabel("Days") 

axs1[1,0].plot(predTnsr[:,:,2])
axs1[1,0].set_title("MicroSensors, High")
axs1[1,0].set_ylabel("Temperature")
axs1[1,0].set_xlabel("Days")

axs1[1,1].plot(predMtrx)
axs1[1,1].set_title("Weather Stations")
axs1[1,1].set_ylabel("Temperature")
axs1[1,1].set_xlabel("Days")        

fig1.tight_layout()

axs2[0,0].plot(xTnsr[:,:,0])
axs2[0,0].set_title("MicroSensors, Low")
axs2[0,0].set_ylabel("Temperature")
axs2[0,0].set_xlabel("Days")

axs2[0,1].plot(xTnsr[:,:,1])
axs2[0,1].set_title("MicroSensors, Med")
axs2[0,1].set_ylabel("Temperature")
axs2[0,1].set_xlabel("Days") 

axs2[1,0].plot(xTnsr[:,:,2])
axs2[1,0].set_title("MicroSensors, High")
axs2[1,0].set_ylabel("Temperature")
axs2[1,0].set_xlabel("Days")

axs2[1,1].plot(yMtrx)
axs2[1,1].set_title("Weather Stations")
axs2[1,1].set_ylabel("Temperature")
axs2[1,1].set_xlabel("Days")  

fig2.tight_layout()

plt.show()
# %% 
