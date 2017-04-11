# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 16:05:26 2017

@author: JamesMichael
"""
import numpy as np
import scipy as sci
from scipy.sparse import csc_matrix
import copy

def varpro2expfun(alpha,t):
    return np.exp(np.reshape(t,(-1,1)).dot(np.reshape(alpha,(1,-1))))

def varpro2dexpfun(alpha,t,i):
    m=t.size
    n=alpha.size
    if (i<0)|(i>=n):
        raise Exception('varpro2dexpfun: i outside of index range for alpha')
    A=csc_matrix((m,n),dtype=complex)
    ttemp=np.reshape(t,(m,1))
    A[:,i]=ttemp*np.exp(alpha[i]*ttemp)
    return A

def varpro2_solve_special(R,D,b):
    A=np.concatenate((R,D),0)
    b=copy.copy(b)
    m,n=R.shape
    ma,na=A.shape
    if (ma!=len(b))|(ma!=(m+n))|(na!=n):
        raise Exception('Something Went Wrong: Input Matrix Dimensions Inconsistant')
    for i in range(n):
        ind=np.array([i]+[m+k for k in range(i+1)])
        u=A[ind,i][:,None]
        sigma=np.linalg.norm(u)
        beta=1.0/(sigma*(sigma+np.abs(u[0])))
        u[0]=np.sign(u[0])*(sigma+abs(u[0]))
        A[ind,i:]+=-(beta*u).dot((u.conj().T.dot(A[ind,i:])))
        b[ind]+=-(beta*u).dot(u.conj().T.dot(b[ind]))
    RA=np.triu(A)[:n,:n]
    return np.linalg.lstsq(RA,b[:n])[0]
    
def checkinputrange(xname,xval,xmin,xmax):
    if xval>xmax:
        print 'Option {:} with value {:} is greater than {:}, which is not recommended'.format(xname,xval,xmin,xmax)
    if xval<xmin:
        print 'Option {:} with value {:} is less than {:}, which is not recommended'.format(xname,xval,xmin,xmax)
class varpro_opts(object):
    def __init__(self,lambda0=1.0,maxlam=52,lamup=2.0,lamdown=2.0,ifmarq=1,maxiter=30,tol=1.0e-6,eps_stall=1.0e-12,iffulljac=1):
        checkinputrange('lambda0',lambda0,0.0,1.0e16)
        checkinputrange('maxlam',maxlam,0,200)
        checkinputrange('lamup',lamup,1.0,1.0e16)
        checkinputrange('lamdown',lamdown,1.0,1.0e16)
        checkinputrange('ifmarq',ifmarq,-np.Inf,np.Inf)
        checkinputrange('maxiter',maxiter,0,1e12)
        checkinputrange('tol',tol,0,1e16)
        checkinputrange('eps_stall',eps_stall,-np.Inf,np.Inf)
        checkinputrange('iffulljac',iffulljac,-np.Inf,np.Inf)
        self.lambda0=float(lambda0)
        self.maxlam=int(maxlam)
        self.lamup=float(lamup)
        self.lamdown=float(lamdown)
        self.ifmarq=int(ifmarq)
        self.maxiter=int(maxiter)
        self.tol=float(tol)
        self.eps_stall=float(eps_stall)
        self.iffulljac=int(iffulljac)
    def unpack(self):
        return self.lambda0,self.maxlam,self.lamup,self.lamdown,self.ifmarq,self.maxiter,self.tol,self.eps_stall,self.iffulljac
    
    
def varpro2(y,t,phi=[],dphi=[],m=[],n=[],iss=[],ia=[],alpha_init=[],opts=[]):
    if opts==[]:
        opts=varpro_opts()
    lambda0,maxlam,lamup,lamdown,ifmarq,maxiter,tol,eps_stall,iffulljac=opts.unpack()
    
    #initialize values
    alpha=alpha_init
    alphas=np.zeros((len(alpha),maxiter)).astype(complex)
    djacmat=np.zeros((m*iss,ia)).astype(complex)
    err=np.zeros((maxiter,))
    res_scale=np.linalg.norm(y,'fro')
    scales=np.zeros((ia,))
    
    phimat=phi(alpha,t)
    [U,S,V]=np.linalg.svd(phimat,full_matrices=False)
    S=np.diag(S);sd=np.diag(S)
    tolrank=m*np.finfo(float).eps
    irank=np.sum(sd>(tolrank*sd[0]))
    U=U[:,:irank]
    S=S[:irank,:irank]
    V=V[:,:irank].T
    b=np.linalg.lstsq(phimat,y)[0]
    res=y-phimat.dot(b)
    errlast=np.linalg.norm(res,'fro')/res_scale
    
    imode=0
    
    for itern in range(maxiter):
        #build jacobian matrix, looping over alpha indices
        for j in range(ia):
            dphitemp=dphi(alpha,t,j).astype(complex)
            djaca=(dphitemp-csc_matrix(U*csc_matrix(U.T.conj()*dphitemp))).dot(b)
            if iffulljac==1:
                #use full expression for jacobian
                djacb=U.dot(np.linalg.lstsq(S,V.T.conj().dot(dphitemp.T.conj().dot(res)))[0])
                djacmat[:,j]=-(djaca.ravel(order='F')+djacb.ravel(order='F'))
            else:
                djacmat[:,j]=-djaca.ravel()
            scales[j]=1.0
            if ifmarq==1:
                scales[j]=np.minimum(np.linalg.norm(djacmat[:,j]),1.0)
                scales[j]=np.maximum(scales[j],1.0e-6)
        
        #loop to detemine lambda (lambda gives the levenberg part)
        #pre-compute components which don't depend on step-size (lambda)
        #get pivots and lapack-style qr for jacobian matrix
        
        _,_,_,work,_=sci.linalg.lapack.zgeqp3(djacmat)
        djacout,jpvt,tau,_,_=sci.linalg.lapack.zgeqp3(djacmat,work)
        rjac=np.triu(djacout)
        lwork=sci.linalg.lapack.zunmqr('L','C',djacout,tau,res.ravel(order='F')[:,None],-1)[1]
        rhstop=sci.linalg.lapack.zunmqr('L','C',djacout,tau,res.ravel(order='F')[:,None],lwork)[0]
        
        scalespvt=scales[jpvt-1]
        rhs=np.concatenate((rhstop,np.zeros((ia,1)).astype(complex)),0)
        
        delta0=varpro2_solve_special(rjac,lambda0*np.diag(scalespvt),rhs)  
        delta0=delta0[jpvt-1]     
        
        alpha0=alpha.ravel()-delta0.ravel()
        
        phimat=phi(alpha0,t)
        b0=np.linalg.lstsq(phimat,y)[0]
        res0=y-phimat.dot(b0)
        err0=np.linalg.norm(res0,'fro')/res_scale
            
        
        #check if this is an improvement
        if err0<errlast:
            #see if smaller lambda is better
            lambda1=lambda0/lamdown
            delta1=varpro2_solve_special(rjac,lambda1*np.diag(scalespvt),rhs)  
            delta1=delta1[jpvt-1]
            
            alpha1=alpha.ravel()-delta1.ravel()
            phimat=phi(alpha1,t)
            b1=np.linalg.lstsq(phimat,y)[0]
            res1=y-phimat.dot(b1)
            err1=np.linalg.norm(res1,'fro')/res_scale
            
            if err1<err0:
                lambda0=copy.copy(lambda1)
                alpha=copy.copy(alpha1)
                errlast=copy.copy(err1)
                b=copy.copy(b1)
                res=copy.copy(res1)
            else:
                alpha=copy.copy(alpha0)
                errlast=copy.copy(err0)
                b=copy.copy(b0)
                res=copy.copy(res0)
        else:
            #if not, increase lambda until something works
            #this makes the algorithm more like gradient descent
            for j in range(maxlam):
                lambda0=lambda0*lamup
                delta0=varpro2_solve_special(rjac,lambda0*np.diag(scalespvt),rhs)  
                delta0=delta0[jpvt-1]
                
                alpha0=alpha.ravel()-delta0.ravel()
                phimat=phi(alpha0,t)
                b0=np.linalg.lstsq(phimat,y)[0]
                res0=y-phimat.dot(b0)
                err0=np.linalg.norm(res0,'fro')/res_scale
                
                if err0<errlast:
                    break
            if err0<errlast:
                alpha=copy.copy(alpha0)
                errlast=copy.copy(err0)
                b=copy.copy(b0)
                res=copy.copy(b0)
            else:
                #no appropriate step length found
                niter=itern
                err[itern]=errlast
                imode=4
                print 'Failed to find appropriate step length at iteration {:}\n Current residual {:}'.format(itern,errlast)
                return b,alpha,niter,err,imode,alphas
        
        alphas[:,itern]=alpha
        err[itern]=errlast
        
        if errlast<tol:
            #tolerance met
            niter=itern
            return b,alpha,niter,err,imode,alphas
        if itern>0:
            niter=itern
            imode=8
            print 'stall detected: residual reduced by less than {:} \n times residual at previous step. \n iteration {:} \n current residual {:}'.format(eps_stall,itern,errlast)
            return b,alpha,niter,err,imode,alphas
        
        phimat=phi(alpha,t)
        [U,S,V]=np.linalg.svd(phimat,full_matrices=False)
        S=np.diag(S);sd=np.diag(S)
        tolrank=m*np.finfo(float).eps
        irank=np.sum(sd>(tolrank*sd[0]))
        U=U[:,:irank]
        S=S[:irank,:irank]
        V=V[:,:irank].T
    #only get here if failed to meet tolerance in maxiter steps
    niter=maxiter
    imode=1
    print 'failed to reach tolerance after maxiter={:} iterations \n current residual {:}'.format(maxiter,errlast)
        
                          
        
        
        

    
if __name__=='__main__':
   pass
    
    
    ###----OUTPUT ADDITIONAL STUFF IF REQUESTED TO...
    ##---
