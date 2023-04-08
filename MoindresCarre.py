#author Gerbaud FLorent
#Algorithme Moindre Carré
#03/03/2023

import numpy as np
import math
import matplotlib.pyplot as plt
from copy import deepcopy

def MoindresCarres(x,y,deg):
    n=x.size
    B=np.zeros((n,deg+1)) # we define the matrix B which contains the base of polynome
    for i in range(0,n):
        for j in range(0,deg+1): #on met deg+1 car on a le polynome 0 jusqu'a deg donc deg+1
            B[i,j]=x[i]**j #construction de la matrice B
    BtB=np.dot(np.transpose(B),B) # we apply the formula to calcul U^*
    Bty=np.dot(np.transpose(B),y)
    sol=np.dot(np.linalg.inv(BtB),Bty)
    return sol

#__________________ Etude des Emissions mondiales de CO2 du au énergies fosile __________________#

deg=4 # we use Moindre Carré with polynome of degrees 4
dataCO2=np.genfromtxt('EmissionCO2.csv',delimiter=',')
x=np.copy(dataCO2[:,0]) # we get the year
y=np.copy(dataCO2[:,1]) # we get the emission
sol=MoindresCarres(x,y,deg) #value of the vect Uj

ApproxCO2=np.zeros(x.size) #we initialisze the sol of the polynome to 0
for i in range(0,deg+1):
    ApproxCO2=ApproxCO2+sol[i]*x**i # we apply the formula 

plt.figure("CO2")
plt.plot(x,ApproxCO2,x,y)
plt.title('Emissions CO2')
plt.ylabel('Total emission CO2 (MTonnes)')
plt.xlabel('Dates (années)')

#__________________ Etudes des écarts de températures  __________________#

deg=3
dataTemp=np.genfromtxt('Temperature-Globe-Ocean-1880-2019.csv',delimiter=',') #we get the data (difference btwin temp and mean temp by year)
x=np.copy(dataTemp[:,0]) # we get the year
y=np.copy(dataTemp[:,1]) # we get the difference betwin the temp
sol=MoindresCarres(x,y,deg) #we get the sol to apply the polynome the vect Uj
t=np.arange(dataTemp[0,0],2100) # we create a vect from 1800 to 2100 to plot the curve of the difference btwin temp

ApproxTemp=np.zeros(t.size) # we define a vect of 0 which will contains the solutions for each year
for i in range(0,deg+1):
    ApproxTemp=ApproxTemp+sol[i]*t**i # we apply the formula

plt.figure("Temperature")
plt.plot(t,ApproxTemp,x,y)
plt.title('Ecart de température')
plt.ylabel('Ecart des température en C°')
plt.xlabel('Dates (années)')
plt.show()

#__________________ Exercice  __________________#

# x=np.array([[-1.0],[0.0],[1.0],[2.0]])
# y=np.array([[1.0],[0.0],[1.0],[2.0]])
# sol=MoindresCarres(x,y,2)
# print(sol)