import numpy as np

def relu(x):
    return (x>0)*x
def sigmoid(x):
    return 1/(1+np.exp(x))

class Bp:
    #bp neural network for binary classification
    def __init__(self,epoch,nodes,learning_rate,weight={}):
    # nodes includes the input layer and output layer
    #example  nodes=[10,12,12,1] means a single sample.size=10
        self.epoch=epoch
        self.nodes=nodes
        self.learning_rate=learning_rate
        self.weight=weight

    def compute_cost(self,output,Y):
        cost = 1 / 2 * np.sum(np.sum((Y - output) * (Y - output)))
        return cost

    def forward(self,X):
        # column is sample
        result={}
        result['layer0'] = X
        if len(self.weight) == 0:
            for layer in range(1,len(self.nodes)):
                self.weight['W'+str(layer)]=np.random.random((self.nodes[layer-1],\
                                                     self.nodes[layer]))
                self.weight['b'+str(layer)]=np.random.random((self.nodes[layer],1))
        for layer in range(1, len(self.nodes)):
            result['layer'+str(layer)]=sigmoid(np.dot(self.weight['W'+str(layer)].T,\
                                              result['layer'+str(layer-1)])+self.weight['b'+str(layer)])
        return result

    def backward(self,result,Y):
        # result(the output of forward)
        output=result['layer'+str(len(self.nodes)-1)]
        cost=1/2*np.sum(np.sum((Y-output)*(Y-output)))
        cost_diff=-(Y-output)
        grad={}
        grad['W0']=0
        for layer in range(1,len(self.nodes)):
            grad['W'+str(layer)]=cost_diff*output*(1-output)
            grad['b'+str(layer)]=cost_diff*output*(1-output)
            for layer_ in range(len(self.nodes)-1,layer,-1):
                grad['W' + str(layer)]=np.dot(self.weight['W'+str(layer_)],grad['W'+str(layer)])*\
                                           result['layer'+str(layer_-1)]*(1-result['layer'+str(layer_-1)])
                grad['b' + str(layer)] = np.dot(self.weight['W' + str(layer_)], grad['b' + str(layer)])*\
                                         result['layer' + str(layer_-1 )] * (1 - result['layer' + str(layer_-1 )])
            grad['W'+str(layer)]=np.dot(result['layer'+str(layer-1)],grad['W'+str(layer)].T)
        for layer in range(1, len(self.nodes)):
            self.weight['W'+str(layer)]=self.weight['W'+str(layer)]+self.learning_rate*grad['W'+str(layer)]
            self.weight['b'+str(layer)]=self.weight['b'+str(layer)]+self.learning_rate*grad['b'+str(layer)]

    def train(self,X,Y):
        for epoch in range(1,self.epoch+1):
            result=self.forward(X)
            self.backward(result,Y)
            if epoch%100==0:
                cost=self.compute_cost(result['layer'+str(len(self.nodes)-1)],Y)
                print('cost: '+str(cost))
                # _,acc=self.predict(X,Y)
                # print(_)
                # print('acc: '+str(acc))

    def predict(self,X,Y=0):
        result=self.forward(X)
        output = result['layer' + str(len(self.nodes) - 1)]
        # print(output)
        for i in range(0,output.shape[1]):
            for j in range(0,output.shape[0]):
                if output[j,i]==np.max(output[:,i]):
                    output[j,i]=1
                else:
                    output[j,i]=0
        if type(Y)==int:
            return(np.argmax(output,axis=0))
        else:
            r=Y-output
            num=0
            for i in range(0,output.shape[1]):
                if np.max(r[:,i])==0 and np.min(r[:,i])==0:
                    num=num+1
            acc=num/output.shape[1]
            return(acc,np.argmax(output,axis=0))








if __name__=='__main__':
    a=Bp(50,[2,4,3,2],0.01)
    a.train([[0,0,1,1],[0,1,0,1]],[[0,1,1,1],[1,0,0,0]])
    a.predict([[0,0,1,1],[0,1,0,1]],[[0,1,1,1],[1,0,0,0]])