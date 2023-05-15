import numpy as np
# import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class conv_ART:
    def __init__(self,input_size,kernel_size,stride,out_channels,pad,pad_value=0,rho=0.999830,alpha=0.00001,beta=0.01,name=None):

        self.device="cuda"
        self.input_h=input_size[0]
        self.input_w=input_size[1]
        self.in_channels=input_size[2]
        self.kernel_size=kernel_size
        self.stride=stride
        self.out_channels=out_channels
        self.pad=pad

        self.h=(self.input_h-self.kernel_size[0]+2*pad[0])//self.stride[0]+1
        self.w=(self.input_w-self.kernel_size[1]+2*pad[1])//self.stride[1]+1
        
        self.input_src=torch.zeros((self.input_h+pad[0]*2,self.input_w+pad[1]*2,self.in_channels)).to(self.device)
        self.input_temp=torch.zeros((self.h,self.w,kernel_size[0],kernel_size[1],self.in_channels)).to(self.device)

        if pad[0]>0:
            self.input_src[0:pad[0],:,:]=pad_value
            self.input_src[self.input_h+pad[0]::,:,:]=pad_value
        if pad[1]>0:
            self.input_src[:,0:pad[1],:]=pad_value
            self.input_src[:,self.input_w+pad[1]::,:]=pad_value

        self.conv_temp_h=[]
        self.conv_temp_w=[]
        for i in range(self.h):
            for j in range(self.w):
                self.conv_temp_h.append(i*self.stride[0])
                self.conv_temp_w.append(j*self.stride[0])
        self.conv_temp_h=np.array(self.conv_temp_h)
        self.conv_temp_w=np.array(self.conv_temp_w)

        x_size=self.in_channels*kernel_size[0]*kernel_size[1]
        c_max=self.out_channels

        self.name=name
        
        self.layer=fuzzy_ART(x_size, c_max, rho, alpha, beta)

    def train(self,input_feature, input_type="cpu"):
        self.zero_num=0
        if input_type=="cpu":
            self.input_src[self.pad[0]:self.input_h+self.pad[0],self.pad[1]:self.input_w+self.pad[1],:]=torch.from_numpy(input_feature).to(self.device)
        else:
            self.input_src[self.pad[0]:self.input_h+self.pad[0],self.pad[1]:self.input_w+self.pad[1],:]=input_feature
        for i in range(self.kernel_size[0]):
            for j in range(self.kernel_size[1]):
                temp=self.input_src[self.conv_temp_h+i,self.conv_temp_w+j,:]
                self.input_temp[:,:,i,j,:]=temp.reshape(self.h,self.w,self.in_channels)

        X=self.input_temp.reshape((self.h*self.w,-1))
        '''
        l=1
        for i in range(self.h):
            for j in range(self.w):
                plt.subplot(self.h,self.w,l)
                plt.imshow(X[i*self.w+j,:].reshape((5,5)).cpu(),vmin=0,vmax=1)
                l+=1
        plt.savefig("1.png")
        sys.exit()
        '''

        '''
        if torch.sum(X).cpu()==0:
            self.zero_num+=1
            if random.random()<0.1:
                flag,rate=self.layer.train(X,X_dtype="gpu")
        else:
            flag,rate=self.layer.train(X,X_dtype="gpu")
        '''
        flag,rate=self.layer.train_batch(X,X_dtype="gpu")
        return self.zero_num,self.layer.N,self.layer.rho,flag,rate

    def infer(self,input_feature,input_type="cpu",rtl=False):
        if input_type=="cpu":
            self.input_src[self.pad[0]:self.input_h+self.pad[0],self.pad[1]:self.input_w+self.pad[1],:]=torch.from_numpy(input_feature).to(self.device)
        else:
            self.input_src[self.pad[0]:self.input_h+self.pad[0],self.pad[1]:self.input_w+self.pad[1],:]=input_feature

        for i in range(self.kernel_size[0]):
            for j in range(self.kernel_size[1]):
                temp=self.input_src[self.conv_temp_h+i,self.conv_temp_w+j,:]
                self.input_temp[:,:,i,j,:]=temp.reshape(self.h,self.w,self.in_channels)

        X=self.input_temp.reshape((self.h*self.w,-1))
        out_feature=torch.zeros((self.h,self.w,self.out_channels)).to(self.device)

        out=self.layer.infer(X,X_dtype="gpu",rtl=rtl)
        l=0
        for H in range(self.h):
            for W in range(self.w):
                out_feature[H,W,:]=out[l]
                l+=1
        #plt.savefig("a.png")
        #sys.exit()
        return out_feature

    def save(self):
        self.layer.save_params("./models/"+self.name)

    def load(self):
        self.layer.load_params("./models/"+self.name)


class fuzzy_ART:
    def __init__(self, X_size, c_max, rho, alpha=0.00001, beta=1., self_supervision=True):

        self.device="cpu"
        self.M = X_size    # input vector size
        self.c_max = c_max # max categories
        self.rho = rho     # vigilance parameter
        self.alpha = torch.tensor([alpha]).to(self.device)  # choice parameter
        self.beta = torch.tensor([beta]).to(self.device)   # learning rate
        self.self_supervision=self_supervision
        
        self.N = 0         # no. of categories initialized to zero
        self.W = torch.ones( (c_max, self.M*2) ).to(self.device) # initialize weigts with 1s
        self.I_sum=torch.tensor([X_size]).to(self.device)
    
    def complement_code(self,X,device,X_dtype="cpu"):
        if X_dtype=="cpu":
            I = X.to(device)
            # I=torch.from_numpy(X).to(device)
        # I = torch.hstack((X, 1-X))
        return I

    def train(self, X, X_dtype="cpu"):
        I = self.complement_code(X,self.device,X_dtype)   # shape of X = Mx1, shape of I = 2Mx1  #[batch,feature_num]
        match_num=0
        batch_num=np.array(I.shape)[1]
        for batch_id in range(batch_num):
            A=I[batch_id]       
            #A55                             #[cmax,feature_num]
            xa_mod=torch.sum(torch.minimum(A, self.W),dim=1)  
            T=xa_mod/(self.alpha + torch.sum(self.W, dim=1))       #[cmax]
            vigilance=xa_mod / (self.I_sum + self.alpha)            #[cmax]  
            
            J_list=T.sort(descending=True).indices.cpu().tolist()
            match_flag=False
            for J in J_list:
                # Checking for resonance ---
                d = vigilance[J].cpu()
                if d >= self.rho: # resonance occured
                    if self.self_supervision:
                        input = torch.minimum(A, self.W[J,:])
                        self.W[J,:] = self.beta*input + (1-self.beta)*self.W[J,:] # weight update
                    match_flag=True
                    match_num+=1
                    break
            #print(batch_id,match_flag,self.N,match_num)
            if match_flag:
                continue
            if self.N < self.c_max:    # no resonance occured, create a new category
                k = self.N
                self.W[k,:] = A
                self.N += 1
                match_num+=1
            else:
                # self.rho-=self.rho/10.
                print("ART memory over! ",self.rho)
        
        if match_num==batch_num:
            return True,match_num*1.0/batch_num
        return False,match_num*1.0/batch_num

    def train_batch(self, X, X_dtype="cpu"):
        #N=self.N
        I = self.complement_code(X,self.device,X_dtype)   # shape of X = Mx1, shape of I = 2Mx1
        
        I_in=torch.unsqueeze(I,dim=1).repeat(1,self.N+1,1)  #[batch,cmax,feature_num]
        xa_mod=torch.sum(torch.minimum(I_in,self.W),dim=2)  #[batch,cmax]
        T=xa_mod/(self.alpha+torch.sum(self.W,dim=1))  #[batch,cmax]
        vigilance=xa_mod/(self.I_sum+self.alpha) #[batch,cmax]  torch.sum(I,dim=1).unsqueeze(dim=1).repeat(1,self.c_max)
        #print(I,torch.sum(I,dim=1).unsqueeze(dim=1).repeat(1,self.c_max))
        
        J_list=T.sort(dim=-1,descending=True).indices.cpu().tolist()
        #print(np.array(J_list).shape,J_list)
        match_num=0
        for batch_id in range(len(J_list)):
            match_flag=False
            #print(batch_id)
            for J in J_list[batch_id]:
                # Checking for resonance ---
                d = vigilance[batch_id][J].cpu()
                if d >= self.rho: # resonance occured
                    if self.self_supervision:
                        self.W[J,:] = self.beta*I[batch_id] + (1-self.beta)*self.W[J,:] # weight update
                    match_flag=True
                    match_num+=1
                    break
            #print(batch_id,match_flag,self.N,match_num)
            if match_flag:
                continue

            if self.N < self.c_max:    # no resonance occured, create a new category
                k = self.N
                self.W[k,:] = I[batch_id]  #self.beta*I[batch_id] + (1-self.beta)*self.W[k,:]
                self.N += 1
                match_num+=1
            else:
                self.rho-=self.rho/10.
                print("ART memory over! ",self.rho)


            '''
            if N < self.c_max:    # no resonance occured, create a new category
                k = self.N
                self.W[k,:] = self.beta*I[batch_id] + (1-self.beta)*self.W[J,:]
                N += 1
                match_num+=1
            '''

        '''
        if self.N<N:
            self.N+=1
        if self.N>=self.c_max:
            self.rho-=self.rho/10.
            print("ART memory over! ",self.rho)
        '''
        
        if match_num==len(J_list):
            return True,match_num*1.0/len(J_list)
        return False,match_num*1.0/len(J_list)
    
    def infer(self, X, X_dtype="cpu",rtl=False): # rtl : real-time learning (learning while inferring)
        I = self.complement_code(X,self.device,X_dtype) 

        I_in=torch.unsqueeze(I,dim=1).repeat(1,self.c_max,1) #[batch,cmax,feature_num]
        xa_mod=torch.sum(torch.minimum(I_in,self.W),dim=2)  #[batch,cmax]
        T=xa_mod/(self.alpha+torch.sum(self.W,dim=1))  #[batch,cmax]
        if rtl:
            vigilance=xa_mod/(self.I_sum+self.alpha) #[batch,cmax] torch.sum(I,dim=1).unsqueeze(dim=1).repeat(1,self.c_max)
        #J_list=T.sort(dim=-1,descending=True).indices.cpu().tolist()
        
        
        if not rtl:
            out=T
        else:
            self.rho_gpu=torch.tensor([self.rho]).to(self.device)
            Y=(vigilance>self.rho_gpu).float()
            out=T*Y

        '''
        out=torch.zeros((len(J_list),self.c_max)).to(self.device)
        for batch_id in range(len(J_list)):
            if not rtl: # only infer
                for J in J_list[batch_id]:
                    out[batch_id][J]=T[batch_id][J]
            else:       # infer AND learn
                for J in J_list[batch_id]:
                    # Checking for resonance ---
                    d = vigilance[batch_id][J].cpu()
                    if d >= self.rho: # resonance occured
                        #if self.self_supervision:
                        #    self.W[J,:] = self.beta*I[batch_id] + (1-self.beta)*self.W[J,:] # weight update (learning)
                        out[batch_id][J]=T[batch_id][J]
                        #break
        '''
        return out  # return the maximum activated weights anyway
         
    def save_params(self, name): # save weights and no. of categories
        tensro_dict = {'W': self.W,"rho":torch.tensor([self.rho])}
        torch.save(tensro_dict, name)
            
    def load_params(self, file_path):
        tensro_dict = torch.load(file_path)
        self.W=tensro_dict['W']
        self.rho=tensro_dict['rho'].cpu().numpy()[0]

class fuzzy_ARTMAP:
    def __init__(self, X_size, label_size, c_max_a, rho_a, rho_ab, alpha=0.00001, beta=1):
        self.M_a = X_size    # input vector size
        self.M_ab = label_size    # input label vector size
        
        self.c_max_a = c_max_a # max categories for ART-a
        
        self.device="cuda"
        self.rho_a = rho_a    # vigilance parameter for ART-a
        self.rho_a_baseline = rho_a
        self.rho_ab = torch.tensor([rho_ab]).to(self.device)  # vigilance parameter for map field
        self.alpha = torch.tensor([alpha]).to(self.device) # choice parameter
        self.M=torch.tensor([self.M_a]).to(self.device)
        self.beta = torch.tensor([beta]).to(self.device)   # learning rate
        
        self.N_a = 0         # no. of categories of ART_a initialized to zero
        
        self.W_a = torch.ones( (c_max_a, self.M_a*2) ).to(self.device) # initialize W_a with 1s
        self.W_ab = torch.ones( (self.M_ab, c_max_a) ).to(self.device) # initialize W_ab with 1s
        

        self.mem_out=0
    
    def complement_code(self,X,device,X_dtype="cpu"):
        if X_dtype=="cpu":
            X=torch.from_numpy(X).to(device)
        I = torch.hstack((X, 1-X))
        return I

    def train(self, X, one_hot_label,X_dtype="cpu"):
        self.A = self.complement_code(X,self.device,X_dtype)   # shape of X = Mx1, shape of I = 2Mx1
        B = one_hot_label

        self.rho_a =  self.rho_a_baseline

        xa_mod=torch.sum(torch.minimum(self.A,self.W_a),dim=1)
        T=xa_mod/(self.alpha+torch.sum(self.W_a,dim=1))
        vigilance=xa_mod/(torch.sum(self.A)+0.0000001)


        T_ids=T.sort(descending=True).indices.cpu().tolist()

        for J in T_ids:
            while vigilance[J].cpu()>self.rho_a: #match
                K=np.argmax( B )
                if self.W_ab[K,J].cpu()>0 and torch.sum(self.W_ab[:,J]).cpu()==1:
                    #self.W_a[J] = self.beta*torch.minimum(A,self.W_a[J]) + (1-self.beta)*self.W_a[J]
                    self.W_a[J] = self.beta*self.A + (1-self.beta)*self.W_a[J]
                    self.W_ab[:,J] = 0
                    self.W_ab[K,J] = 1
                    return True
                else:
                    self.rho_a=vigilance[J].cpu()

        if self.N_a < self.c_max_a:
            n = self.N_a
            self.W_a[n,:]=self.A
            self.N_a += 1
            
            K = np.argmax( B )
            self.W_ab[:,n] = 0
            self.W_ab[K,n] = 1
            return True
        else:
            self.mem_out+=1
            return False

    
    def infer(self, X, X_dtype="cpu"):
        A = self.complement_code(X,self.device,X_dtype)
        xa_mod=torch.sum(torch.minimum(A,self.W_a),dim=1)
        T=xa_mod/(self.alpha+torch.sum(self.W_a,dim=1))
        #T=torch.sum(torch.minimum(A,self.W_a),dim=1) + ((1-self.alpha)*(self.M-torch.sum(self.W_a,dim=1)))
        #vigilance=torch.sum(torch.minimum(A,self.W_a),dim=1)/self.M

        T_ids=T.sort(descending=True).indices.cpu().tolist()
        J_max=T_ids[0]

        xab=self.W_ab[:,J_max]

        return xab.cpu()

    def save(self):
        tensro_dict = {'W_a': self.W_a, 'W_ab': self.W_ab}
        torch.save(tensro_dict, './models/fuzzyart')

    def load(self):
        tensro_dict = torch.load('./models/fuzzyart')
        self.W_a=tensro_dict['W_a']
        self.W_ab=tensro_dict['W_ab']

if __name__=='__main__':
    '''
    model=fuzzy_ART(X_size=3, c_max=5, rho=0.1)
    a=np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
    print(model.train(a))
    print(model.infer(a))
    '''
    model=conv_ART(input_size=(100,100,1), kernel_size=(5,5),stride=(2,2),out_channels=32,pad=(0,0))

    test_data = pd.read_csv("datasets/MNIST/mnist_test.csv").values
    test_data = test_data[:-1, :]
    test_data = test_data[:,1:]
    test_data = test_data.reshape(-1, 28, 28, 1)

    for i in range(1000):
        _,_,_,_,r=model.train( test_data[i]/255. )
        print(r)
    #model.infer( test_data[0] )

