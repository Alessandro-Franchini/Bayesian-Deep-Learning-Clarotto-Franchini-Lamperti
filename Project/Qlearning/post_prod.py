##############################  BAYESIAN DEEP LEARNING  ############################## 
############### Clarotto Lucia, Franchini Alessandro, Lamperti Letizia ############### 

############### Q-LEARNING CARTPOLE PROBLEM  ###############
###############     Analysis of results      ###############
###############    Plots and comparisons     ###############
 


######### LIBRARIES #########

import os
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import time
import pandas as pd

# Path for plots
my_path = os.path.dirname(__file__)+ '/img/'



######### POST-PROCESSING #########

df = pd.read_csv('file_df_3.csv')
df['Angle']= df['Angle']*180/np.pi
Pos = df['Pos']
Vel = df['Vel']
Angle = df['Angle']
Angle_rad= Angle/180*np.pi
Vel_tip = df['Vel_tip']
Mean1 = df['Mean_1']
Mean2 = df['Mean_2']
Var1 = df['Variance_1']
Var2 = df['Variance_2']
t=range(0,df.shape[0])


### Plot of MEANS

plt.figure(figsize=(10,5))
plt.plot(t,Mean1, 'steelblue',alpha=1, label='Mean 1')
plt.plot(t,Mean2, 'orange', alpha=1, label='Mean 2')
plt.title('Means')
plt.legend(loc='upper right')
plt.savefig(my_path+'Means.pdf', bbox_inches = 'tight')
plt.show()


### Plot of VARIANCES

plt.figure(figsize=(10,5))
plt.plot(t,Var1, 'blue',alpha=1, label='Variance 1')
plt.plot(t,Var2, 'red', alpha=1, label='Variance 2')
plt.title('Variances')
plt.legend(loc='upper right')
plt.savefig(my_path+'Variances.pdf', bbox_inches = 'tight')
plt.show()


### Plot of STATE

plt.figure(figsize=(10,5))
plt.plot(t,Pos, 'turquoise', alpha=1, label='Position')
#angle in radiant
plt.plot(t,Angle_rad, 'orchid', alpha=1, label='Angle')
#plt.plot(t,Angle, 'red', alpha=1, label='Angle')
plt.plot(t,Vel, 'green', alpha=1, label='Velocity')
plt.plot(t,Vel_tip, 'yellowgreen', alpha=1, label='Velocity tip')
plt.title('State')
plt.legend(loc='upper right')
plt.savefig(my_path+'State.pdf', bbox_inches = 'tight')
plt.show()

### Comparison ANGLE and VARIANCES

plt.figure(figsize=(10,5))
plt.plot(t,Angle_rad, 'orchid',alpha=1, label='Angle')
plt.plot(t,Var1, 'blue',alpha=1, label='Variance 1')
plt.plot(t,Var2, 'red', alpha=1, label='Variance 2')
plt.title('Angle-Variances')
plt.legend(loc='upper right')
plt.savefig('Angle_Variances.pdf', bbox_inches = 'tight')
plt.show()

### Comparison POSITION and VARIANCES

plt.figure(figsize=(10,5))
plt.plot(t,Pos, 'turquoise',alpha=1, label='Position')
plt.plot(t,Var1, 'blue',alpha=1, label='Variance 1')
plt.plot(t,Var2, 'red', alpha=1, label='Variance 2')
plt.title('Variances')
plt.legend(loc='upper right')
plt.savefig(my_path+'Position_Variances.pdf', bbox_inches = 'tight')
plt.show()



###### If you want to try with restricted data ########

#plt.figure(figsize=(10,5))
#plt.plot(t[200:220],Mean1[200:220], 'steelblue',alpha=1, label='Mean 1')
#plt.plot(t[200:220],Mean2[200:220], 'orange', alpha=1, label='Mean 2')
#plt.title('Means')
#plt.legend(loc='upper right')
#plt.savefig(my_path+'Means_restricted.pdf', bbox_inches = 'tight')
#plt.show()
#
#plt.figure(figsize=(10,5))
#plt.plot(t[200:220],Var1[200:220], 'blue',alpha=1, label='Variance 1')
#plt.plot(t[200:220],Var2[200:220], 'red', alpha=1, label='Variance 2')
#plt.title('Variances')
#plt.legend(loc='upper right')
#plt.savefig(my_path+'Variances.pdf', bbox_inches = 'tight')
#plt.show()

#plt.figure(figsize=(10,5))
##plt.plot(t[200:260],Mean1[200:260], 'steelblue',alpha=1, label='Mean 1')
#plt.plot(t[200:260],Mean1[200:260]+np.sqrt(Var1[200:260]), 'darkblue', alpha=1, label='Mean 1 +/- Std 1')
#plt.plot(t[200:260],Mean1[200:260]-np.sqrt(Var1[200:260]), 'darkblue', alpha=1)
##plt.title('Mean +/- Std')
##plt.legend(loc='upper right')
##plt.show()
#
##plt.figure(figsize=(10,5))
##plt.plot(t[200:260],Mean2[200:260], 'orange',alpha=1, label='Mean 2')
#plt.plot(t[200:260],Mean2[200:260]+np.sqrt(Var2[200:260]), 'orangered', alpha=1, label='Mean 2 +/- Std 2')
#plt.plot(t[200:260],Mean2[200:260]-np.sqrt(Var2[200:260]), 'orangered', alpha=1)
#plt.title('Mean +/- Std')
#plt.legend(loc='upper right')
#plt.savefig(my_path+'Comparisons_restricted.pdf', bbox_inches = 'tight')
#plt.show()


### Try to wee if VARIANCE change when the angle is next to 0
## Seems to be low variance

df_top_angles = df.loc[(df['Angle'] >= -0.1) & (df['Angle'] <= 0.1)]
time = range(0,df_top_angles.shape[0])
plt.figure(figsize=(10,5))
#plt.plot(time,df_top_angles.Mean_1, 'steelblue',alpha=1, label='Mean 1')
#plt.plot(time,df_top_angles.Mean_1+np.sqrt(df_top_angles.Variance_1), 'darkblue', alpha=1, label='Mean 1 +/- Std 1')
#plt.plot(time,df_top_angles.Mean_1-np.sqrt(df_top_angles.Variance_1), 'darkblue', alpha=1)
##plt.plot(time,df_top_angles.Mean_2, 'orange',alpha=1, label='Mean 2')
#plt.plot(time,df_top_angles.Mean_1+np.sqrt(df_top_angles.Variance_2), 'orangered', alpha=1, label='Mean 2 +/- Std 2')
#plt.plot(time,df_top_angles.Mean_1-np.sqrt(df_top_angles.Variance_2), 'orangered', alpha=1)
plt.plot(time,df_top_angles.Variance_1, 'blue',alpha=1, label='Var 1')
plt.plot(time,df_top_angles.Variance_2, 'red',alpha=1, label='Var 2')
plt.title('Variance for top angles')
plt.legend(loc='upper right')
#plt.savefig(my_path+'Top_angles.pdf', bbox_inches = 'tight')
plt.show()

### Try to wee if VARIANCE change when the angle is far from 0
## Seems to be HIGH variance

df_low_angles = df.loc[(df['Angle'] >= 6) | (df['Angle'] <= -6)]
time = range(0,df_low_angles.shape[0])
plt.figure(figsize=(10,5))
plt.plot(time,df_low_angles.Variance_1, 'blue',alpha=1, label='Var 1')
plt.plot(time,df_low_angles.Variance_2, 'red',alpha=1, label='Var 2')
plt.title('Variance for low angles')
plt.legend(loc='upper right')
#plt.savefig(my_path+'Low_angles.pdf', bbox_inches = 'tight')
plt.show()

