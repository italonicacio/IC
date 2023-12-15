import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report


def funcObjetivo(c,z,data):
    N,n = data.shape
    J = 0
    for i in range(N):
        J = J + np.linalg.norm(data[i,:]-z[c[i],:])**2
    return J/N

def agrupar(z,k,data):
    N,n = data.shape
    c = np.zeros(N,dtype=int)
    for i in range(N):
        distancias= np.zeros(k)
        for j in range(k):
            distancias[j] = np.linalg.norm(z[j,:]-data[i,:])
        c[i] = np.argmin(distancias)
    for i in range(k):
        if np.sum(c==i)==0: # verifica se todos os grupos tem pelo menos um vetor
            print('Grupo',i,'não tem elementos.')
            assert 0
    return c

def representantes(c,k,data):
    N,n = data.shape
    z = np.zeros((k,n))
    for i in range(k):
        onde = np.where(c==i)
        onde = np.squeeze(onde)
        Xk = data[onde,:]
        Nk = Xk.shape[0]
        z[i,:] = np.sum(Xk, axis = 0)
        z[i,:] = z[i,:]/Nk
    return z

def sorteio_representantes(k,data):
    N,n=data.shape
    while True:
        r = np.random.randint(0,N,k)
        r.sort()
        b = r[1:]
        a = r[:-1]
        c = a < b
        if np.sum(c)==len(c):
            break
    z = data[r,:]
    return z,r


def kmeans01(data, k, maxiter, epsilon, sortear = True, r = 0):
    N,n = data.shape
    J = np.zeros(maxiter)
    
    if sortear: # inicializa com z sorteado a partir dos vetores em data
        z,r = sorteio_representantes(k,data)
    else:
        z=data[r,:]

    c = agrupar(z,k,data)
    J[0] = funcObjetivo(c,z,data)
    z=representantes(c,k,data)
    
    for i in range(1,maxiter):
        c = agrupar(z,k,data)
        J[i] = funcObjetivo(c,z,data)
        print('iter=',i,'\t\tJ=',J[i])
        if J[i]==J[i-1]: #abs(J[i]-J[i-1])/J[i]<epsilon:
            J=J[0:i+1]
            print('kmeans converge em',i,'iterações e J*=',J[i],'.')
            return r,c,J
        z=representantes(c,k,data)
        
    print('kmeans atinge o número máximo de iterações',maxiter,'.')
    return r,c,J



data, targets = load_wine(return_X_y = True)

targets = np.array(targets)
data = np.array(data)


# %%
N,n=data.shape
print('Classe 0: ', len(np.where(targets==0)[0]))
print('Classe 1: ', len(np.where(targets==1)[0]))
print('Classe 2: ', len(np.where(targets==2)[0]))


print('No total são ', N, 'vinhos.')


# normalizar = True
# X = data
# if normalizar:
#     maxdata = np.max(data[0])
#     X = X/maxdata

X = scale(data)

k = 3
maxiter = 100
epsilon = 1e-6

Jmin = 1e30
rmin = np.zeros(k)
for i in range(k):
    print('rodada=',i)
    r, c, J = kmeans01(X, k, maxiter, epsilon)
    if min(J) < Jmin:
        Jmin = np.min(J)
        rmin = r
    

r, c, J = kmeans01(X, k, maxiter, epsilon, sortear = False, r = rmin)


print(classification_report(targets,c))

print('Agrupamento do KMeans:')
print('Quantidade da classe 0: ', np.count_nonzero(c == 0))
print('Quantidade da classe 1: ', np.count_nonzero(c == 1))
print('Quantidade da classe 2: ', np.count_nonzero(c == 2))




