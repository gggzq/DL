import numpy as np
from sklearn.cluster import KMeans

class Rbf:
    #Rbf rbf neural network
    def __init__(self,epoch,nodes,lambd,learning_rate,X,weight={}):
        # X  row for sample
        self.epoch=epoch
        self.nodes=nodes
        self.lambd=lambd
        self.learning_rate=learning_rate
        self.weight=weight
        self.centers,self.sigma2=self.hidden_layer_parameters(X)

    def compute_cost(self,output,Y):
        cost = 1 / 2 * np.sum(np.sum((Y - output) * (Y - output)))
        return cost

    def hidden_layer_parameters(self,X):
        # X row for sample
        m=self.nodes[1]
        kmeans=KMeans(n_clusters=m,random_state=0).fit(X)
        centers=kmeans.cluster_centers_
        sigma2=np.zeros((m,1))
        for i in range(0,m):
            sigma2[i]=self.lambd*(np.max((centers-centers[i,:])**2))
        return centers,sigma2

    def forward(self,X):
        result={}
        result['layer0']=X
        m,n=X.shape
        result['layer1']=np.zeros((m,self.nodes[1]))
        for i in range(0,m):
            for j in range(0,self.nodes[1]):
                result['layer1'][i,:]=(np.exp(-np.sum((self.centers-X[i,:].reshape(1,n))**2,1).reshape(1,n)/(2*self.sigma2[j])))
        if len(self.weight)==0:
            self.weight['W']=np.random.random((self.nodes[1],1))
        result['layer2']=np.dot(result['layer1'],self.weight['W'])
        return result

    def backward(self,result,X,Y):
        output=result['layer2']
        m,n=X.shape
        cost=self.compute_cost(output,Y)
        grad={}
        grad['mu']=np.zeros((self.nodes[1],n))
        grad['sigma2']=np.zeros((self.nodes[1],1))
        grad['W']=-np.dot(result['layer1'].T,(Y-output))
        for node in range(0,self.nodes[1]):
            for j in range(0,m):
                grad['mu'][node,:]=grad['mu'][node,:]+\
                                   self.weight['W'][node]/self.sigma2[node]*\
                                   (X[j,:]-self.centers[node,:]).reshape(1,n)*(output[j]-Y[j])*\
                                   np.exp(-np.sum((X[j,:]-self.centers[node,:])**2)/(2*self.sigma2[node]))
                grad['sigma2'][node]=grad['sigma2'][node]+0.5*self.weight['W'][node]/(self.sigma2[node]**2)*(output[j]-Y[j])*np.exp(-np.sum((X[j,:]-self.centers[node])**2)/self.sigma2[node])
        self.weight['W']=self.weight['W']-self.learning_rate*grad['W']
        self.centers=self.centers-self.learning_rate*grad['mu']
        self.sigma2=self.sigma2-self.learning_rate*grad['sigma2']
        return cost
    def train(self,X,Y):
        for epoch in range(0,self.epoch):
            result=self.forward(X)
            cost=self.backward(result,X,Y)
            if epoch%100==0:
                print('cost: ')
                print(cost)
                print('acc: ')
                print(self.predict(X,Y))

    def predict(self,test_data,y=0):
        result=self.forward(test_data)
        result=(result['layer2']>=0.5)*1
        if type(y)==int:
            pass
            return(result)
        else:
            r=y-result
            num=np.sum((r==0)*1)
            return(num/r.shape[0])















if __name__=='__main__':
    Rbf(100,2,2)