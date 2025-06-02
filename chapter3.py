import cv2
import numpy as np

L= 256


def Negative(imgin):
    # M la do cao cuar anh
    # N la do rong cua anh
    # anh la ma tran m hang N Cot (MxN)
    M,N =imgin.shape
    imgout = np.zeros((M,N),np.uint8) 
    for x in range (0,M):
        for y in range (0,N):
            r =imgin[x,y]
            s=L -1-r
            imgout[x,y]=np.uint8(s)
    return imgout 

def NegativeColor(imgin):
    # M la do cao cuar anh
    # N la do rong cua anh
    # anh la ma tran m hang N Cot (MxN)
    M,N,C=imgin.shape
    imgout = np.zeros((M,N,C),np.uint8) 
    for x in range (0,M):
        for y in range (0,N):
            # anh mau cuar opnecv laf BGR
            # anh mau cua pillow la RGB
            # Pillow la thu vien anh cua python
            b = imgin[x,y,0]
            b= L-1-b

            g = imgin[x,y,1]
            g= L-1-g

            r = imgin[x,y,2]
            r= L-1-r

            imgout[x,y,0]=np.uint8(b)
            imgout[x,y,1]=np.uint8(g)
            imgout[x,y,2]=np.uint8(r)
    return imgout            

def Logarit(imgin):
    M,N = imgin.shape
    imgout  = np.zeros((M,N),np.uint8)
    c= (L-1.0)/np.log(1.0*L)
    for x in range (0,M):
        for y in range(0,N):
            r= imgin [x,y]
            if r==0:
                r=1
            s= c*np.log(1.0+r)
            imgout[x,y]= np.uint8(s)
    return imgout

def Power(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N),np.uint8)
    gamma = 5.0
    c = np.power(L-1.0, 1- gamma)
    for x in range (0, M):
        for y in range (0, N):
            r = imgin[x, y]
            if r == 0:
                r = 1
            s = c*np.power(1.0*r, gamma)
            imgout[x,y] = np.uint8(s)
    return imgout

def Piecewiseline(imgin):
    M,N = imgin.shape
    imgout  = np.zeros((M,N),np.uint8) 
    rmin,rmax ,_,_ =cv2.minMaxLoc(imgin)
    r1=rmin
    s1=0
    r2= rmax
    s2=L-1
    for x in range (0,M):
        for y in range(0,N):
            r= imgin[x,y]
            #Doan 1
            if r < r1:
                s=s1*1.0/r1*r
            #doan 2
            elif r< r2:
                s=1.0*(s2-s1)/(r2-r1)*(r-r1)+s1
            #doan 3
            else:
                s=1.0*(L-1-s2)/(L-1-r2)*(r-r2)+s2
            imgout [x,y]=np.uint8(s)
    return imgout    

def Histogram(imgin):
    M,N = imgin.shape
    imgout  = np.zeros((M,N,3),np.uint8) + np.uint8(255)
    h=np.zeros(L,np.int32)
    for x in range (0,M):
        for y in range (0,N):
            r= imgin[x,y]
            h[r]=h[r]+1
    p= 1.0*h/(M*N)
    scale= 3000
    for r in range (0,L):
        cv2.line(imgout,(r,M-11),(r,M-1-np.int32(scale*p[r])),(255,0,0))        
    return imgout  

def HistEqual(imgin):
    M,N = imgin.shape
    imgout  = np.zeros((M,N),np.uint8) 
    h=np.zeros(L,np.int32)
    for x in range (0,M):
        for y in range (0,N):
            r= imgin[x,y]
            h[r]=h[r]+1
    p= 1.0*h/(M*N)
    s= np.zeros(L,np.float64)
    for k in range (0,L):
        for j in range (0,k+1):
            s[k]=s[k]+p[j]
    for x in range (0,M):
        for y in range (0,N):
            r= imgin[x,y]
            imgout[x,y]=np.uint8((L-1)*s[r])
    return imgout  

def LocalHist(imgin):
    M,N = imgin.shape
    imgout  = np.zeros((M,N),np.uint8) 
    m=3 
    n=3
    a = m//2
    b=n//2
    for x in range (a,M-a):
        for y in range (b,N-1):
            w= imgin[x-a:x+a+1,y-b:y+b+1]
            w=cv2.equalizeHist(w)
            imgout [x,y]=w[a,b]
    return imgout

def HistStat(imgin):
    M,N = imgin.shape
    imgout  = np.zeros((M,N),np.uint8) 
    mean,stdev=cv2.meanStdDev(imgin)
    mG= mean[0,0]
    simaG=stdev[0,0]

    m=3 
    n=3

    C=22.8
    k0=0
    k1=0.1
    k2=0
    k3=0.1

    a = m//2
    b=n//2
    for x in range (a,M-a):
        for y in range (b,N-1):
            w= imgin[x-a:x+a+1,y-b:y+b+1]
            mean,stdev=cv2.meanStdDev(w)
            msxy =mean[0,0]
            simasxy= stdev[0,0]
            if (mG*k0 <= msxy<= mG*k1) and (k2*simaG<=simasxy<=k3*simaG):
                imgout[x,y]=np.uint8(C*imgin[x,y])
            else:
                imgout[x,y]=imgin[x,y]   
    return imgout

def Sharp(imgin):
    w=np.array([[1,1,1],[1,-8,1],[1,1,1]],np.float32)
    Laplacian= cv2.filter2D(imgin,cv2.CV_32FC1,w)
    imgout =imgin -Laplacian
    imgout= np.clip(imgout,0,L-1)
    imgout= imgout.astype(np.uint8)
    return imgout

def Gradien(imgin):
    Sobel_x=np.array([[1,2,1],[0,0,0],[-1,-2,-1]],np.float32)
    Sobel_y=np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float32)
    gx= cv2.filter2D(imgin,cv2.CV_32FC1,Sobel_x)
    gy= cv2.filter2D(imgin,cv2.CV_32FC1,Sobel_y)
    imgout = abs(gx)+abs(gy)

    imgout= np.clip(imgout,0,L-1)
    imgout= imgout.astype(np.uint8)

    return imgout


   
       
        
   
             
                