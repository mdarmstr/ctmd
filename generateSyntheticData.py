# GenerateSyntheticData
# The goal of this script is to generate synthetic data that is characteristic of the kind of data we expect to see.
# The data will be arranged according to a PARAFAC model with 3 modes. Xk = BkDkA.T. The first mode will contain the time-series data that is expected to drift between Dk sensors. The A will encapsulate a general trend that is observed yearly.
# For the purposes of this experiment, the weather patterns are going to be idealised gaussians for the macro-climate data. The micro-climate data in this case will be assumed to be data collected from underground. As such, it is reasonable to expect a delay and reduced impact.
#Importing the basic toolboxes.

# %%
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

#General function for constructing a Gaussian
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


# Create a test dataset of 30 days, 10 snrs, 3 levels and 2 weather stations
test = synData(30,10,3,2)    
xTnsr, yMtrx = test.tMat()        

fig, axs = plt.subplots(2,2)

# low microclimate
axs[0,0].plot(xTnsr[:,:,0])
axs[0,0].set_title("MicroSensors, Low")
axs[0,0].set_ylabel("Temperature")
axs[0,0].set_xlabel("Days")

axs[0,1].plot(xTnsr[:,:,1])
axs[0,1].set_title("MicroSensors, Med")
axs[0,1].set_ylabel("Temperature")
axs[0,1].set_xlabel("Days") 

axs[1,0].plot(xTnsr[:,:,2])
axs[1,0].set_title("MicroSensors, High")
axs[1,0].set_ylabel("Temperature")
axs[1,0].set_xlabel("Days")

axs[1,1].plot(yMtrx)
axs[1,1].set_title("Weather Stations")
axs[1,1].set_ylabel("Temperature")
axs[1,1].set_xlabel("Days")        

fig.tight_layout()
plt.show() 

# %%
