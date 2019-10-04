import numpy as np
import random

""" We Require
    Points in d dimensions
    Labels of points
    Kmeans_result
    params:
        iter
        G
        S
        ri
        rj
        alpha
"""
def distance(xi,xj):
    # Xi= 1*D
    # Xj= 1*D
    summ=0
    for d in range(len(xi)):
        summ+=np.power((xi[d]-xj[d]),2)
    return np.sqrt(summ)

def fitness(Points,label,particle):
    fitness=0
    for i in range(len(Points)):
        cluster_no=label[i]
        #print("C",cluster_no, len(Points[i]))
        centroid=particle[int(cluster_no)*len(Points[i]):(int(cluster_no)+1)*len(Points[i])]
        fitness+=distance(Points[i],centroid)**2
    return fitness


def Fitness_for_All(Particles, Labels,Points):
    # Candidate Solution =n*s
    Total_fitness=[]
    for i in range(len(Particles)):
        Total_fitness.append(fitness(Points,Labels,Particles[i]))
    return Total_fitness


def compute_extras(Total_fitness):
    worst=min(Total_fitness)
    #print(worst)
    den=0
    for i in range(len(Total_fitness)):
        den+=abs(Total_fitness[i]-worst)
    #print(den)
    return worst,den

def Mass(Particles,Labels,Points,Total_fitness):
    Masses=[]
    worst,den=compute_extras(Total_fitness)
    for i in range(len(Particles)):
        Masses.append(1/(abs(Total_fitness[i]-worst)+0.01))
    return Masses


def acc(Particles,Masses,rj,G):
    #For every D dimension
    force=np.ndarray(Particles.shape,dtype=np.float64)
    for i in range(len(Particles)):
        Fi=np.ndarray((1,Particles.shape[1]),dtype=np.float64)
        #print(Fi.shape,(1,Particles.shape[1]))
        for j in range(len(Particles)):
            if(j!=i):
                #print(Fi.shape,Particles.shape)
                Fi+=rj*G*Masses[j]*(Particles[i]-Particles[j])/(distance(Particles[i],Particles[j])+0.000001)
        #print(force.shape)
        force[i,:]=Fi
    return force

def velocity(Particles,vel,acc,ri):
    vel=vel*ri+acc
    return vel

def Update(Particles,vel):
    Particles=Particles+vel
    return Particles

def Generate_Solutions(Points,S,d,k,Particle,tk):
    l=Points.min()
    r=Points.max()
    #print(l,r)
    #print(S,d*k)
    np.random.seed(tk)
    Particles=np.random.uniform(float(l),float(r),(int(S),d*k))
    #Particles=np.array([[1,1,-1,-1],[3,2,-3,-2],[2,2,-3,-2],[2.5,3,-3,-2.5],[1,1,-3,2.5]])
    Particles[0,:]=Particle
    return Particles

def UpdateG(G,iter,it,alpha):
    return G*np.exp(-iter*alpha/it)

def init(Points,Kmeans_result,Labels,params):
    #Kmeans_result=d*k size array
    S=params['S']
    d=Kmeans_result.shape[1]
    k=Kmeans_result.shape[0]
    Kmeans_result=Kmeans_result.flatten()
    #print("Kmeans ",Kmeans_result)
    tk=params['seed']
    Particles=Generate_Solutions(Points,S,d,k,Kmeans_result,tk)
    #print(Particles)
    G=params['G']
    Solution=np.array((1,Kmeans_result[0]*Kmeans_result[1]),dtype=np.float64);
    #print(Solution.shape)
    vel=np.zeros(shape=(S,d*k),dtype=np.float64)
    minFitness=10000000;
    #print(random.uniform(0,1))
    #print(random.uniform(0,1))
    #print(random.uniform(0,1))
    #print(random.uniform(0,1))
    #print(Particles[0])
    for i in range(params['iter']):
        Total_fitness=Fitness_for_All(Particles,Labels,Points)
        #print("Fitness ",Total_fitness)
        Masses=Mass(Particles,Labels,Points,Total_fitness);
        print("Masses",Masses)
        Acc=acc(Particles,Masses,params['ri'],G)
        #print("Acceleration",Acc)
        vel=velocity(Particles,vel,Acc,params['rj'])
        #print("Velocity",vel)
        Particles=Update(Particles,vel)
        if(min(Total_fitness)<minFitness):
            for i in range(len(Total_fitness)):
                if(Total_fitness[i]<minFitness):
                    minFitness=Total_fitness[i]
                    Solution=Particles[i,:]
        #print(Particles[0])
        G=UpdateG(params['G'],i,params['iter'],params['alpha'])
    #print(Solution.shape)
    #print(Kmeans_result.shape)
    Solution.resize((k,d))
    newlabels=[]
    #print(Points)
    #print(Solution)
    for i in range(Points.shape[0]):
        minvalue=10000000
        ind=-1
        for j in range(Solution.shape[0]):
            if(distance(Solution[j],Points[i])<minvalue):
                ind=j
                minvalue=distance(Solution[j],Points[i])
        #print(ind)
        newlabels.append(ind)
    #print(newlabels)
    return np.array(newlabels,dtype=np.float64)