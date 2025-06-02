import numpy as np
import cv2
L = 256
def Spectrum(imgin):
    M, N = imgin.shape

    #  buoc 1 va 2: tao anhr co kich thuoc PxQ
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp = np.zeros((P, Q), np.float32)
    fp[:M,:N] = 1.0*imgin/(L-1)

    # buoc 3: nhan fp voi (-1)^(x+y)
    for x in range (0,M):
        for y in range (0,N):
            if (x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]

    # buoc 4: tinh bien doi fourier thuan DFT
    F = cv2.dft(fp, flags = cv2.DFT_COMPLEX_OUTPUT)
    FR = F[:,:,0]
    FI = F[:,:,1]

    # tinh pho
    S = np.sqrt(FR**2 + FI**2)
    S = np.clip(S, 0, L-1)
    imgout = S.astype(np.uint8)
    return imgout
def RemoveMoireSimple(imgin):
    M,N = imgin.shape
    #Buoc 1 vaf 2 : tao anh cos kich thuocw PxQ
    P=cv2.getOptimalDFTSize(M)
    Q= cv2.getOptimalDFTSize(N)
    fp=np.zeros((P,Q),np.float32)
    fp[:M,:N]=1.0*imgin

    # buoc 3:nhan fp với -1^ x+y
    for x in range(0,M):
        for y in range(0,N):
            if (x+y)%2== 1:
                fp[x,y]=-fp[x,y]

    #bước 4
    F=cv2.dft(fp,flags=cv2.DFT_COMPLEX_OUTPUT)
    FR=F[:,:,0]            
    FI=F[:,:,1]
    
    
    # bước 5
    H= CreateNocthFilter(P,Q)
    # buoc 6 nhan G(u,v)=F(u,v)*H(u,v)
    G=cv2.mulSpectrums(F,H,flags=cv2.DFT_ROWS)
    # bươc 7 IDFT
    g =cv2.idft(G,flags=cv2.DFT_SCALE)
    #Bước 8 lấy lại kích thước ban đầu là MxN ,là lấy phần thực,nhân (-1)^(X+Y)
    gR=g[:M,:N,0]
    for x in range(0,M):
        for y in range (0,N):
            if (x+y)%2==1:
                gR[x,y]=-gR[x,y]
    gR=np.clip(gR,0,L-1)
    imgout=gR.astype(np.uint8)           
        


    return imgout

    

def CreateNocthFilter(P,Q):
    H=np.ones((P,Q,2),np.float32)
    H[:,:,1]=0.0
    u1 =45; v1=58
    u2=86; v2 =58
    u3= 41; v3=119
    u4 =83;v4 =119

    u5= P-u1; v5=Q-v1
    u6 =P-u2; v6=Q-v2
    u7=P-u3; v7=Q-v3
    u8=P-u4; v8=Q-v4
    D0=15

    for u in range(0,P):
        for v in range(0,Q):
            Duv=np.sqrt((1.0*u-u1)**2+(1.0*v-v1)**2)
            if Duv <=D0:
                H[u,v,0]=0.0
            Duv=np.sqrt((1.0*u-u2)**2+(1.0*v-v2)**2)
            if Duv <=D0:
                H[u,v,0]=0.0 
            Duv=np.sqrt((1.0*u-u3)**2+(1.0*v-v3)**2)
            if Duv <=D0:
                H[u,v,0]=0.0 
            Duv=np.sqrt((1.0*u-u4)**2+(1.0*v-v4)**2)
            if Duv <=D0:
                H[u,v,0]=0.0 
            Duv=np.sqrt((1.0*u-u5)**2+(1.0*v-v5)**2)
            if Duv <=D0:
                H[u,v,0]=0.0 
            Duv=np.sqrt((1.0*u-u6)**2+(1.0*v-v6)**2)
            if Duv <=D0:
                H[u,v,0]=0.0 
            Duv=np.sqrt((1.0*u-u7)**2+(1.0*v-v7)**2)
            if Duv <=D0:
                H[u,v,0]=0.0 
            Duv=np.sqrt((1.0*u-u8)**2+(1.0*v-v8)**2)
            if Duv <=D0:
                H[u,v,0]=0.0 
    return H
def DrawNotchFilter(imgin):
    M,N = imgin.shape
    #Buoc 1 vaf 2 : tao anh cos kich thuocw PxQ
    P=cv2.getOptimalDFTSize(M)
    Q= cv2.getOptimalDFTSize(N)
    H= CreateNocthFilter(P,Q)
    HR=H[:,:,0]*(L-1)
    imgout =HR.astype(np.uint8)
    return imgout

def CreateNocthFilterRemove(P,Q):
    H=np.ones((P,Q,2),np.float32)
    H[:,:,1]=0.0
    D0=10
    v0= Q//2
    for u in range(0,P):
        for v in range(0,Q):
            if  u not in range(P//2-10,P//2+10+1):
                if abs(v -v0) <= D0:
                    H[u,v,0]=0.0
    return H

def DrawNotchPeriodFilter(imgin):
    M,N = imgin.shape
    P=cv2.getOptimalDFTSize(M)
    Q= cv2.getOptimalDFTSize(N)
    H= CreateNocthFilterRemove(P,Q)
    HR=H[:,:,0]*(L-1)
    imgout =HR.astype(np.uint8)
    return imgout

def RemovePeriodNoise(imgin):
    M,N = imgin.shape
    P=cv2.getOptimalDFTSize(M)
    Q= cv2.getOptimalDFTSize(N)
    fp=np.zeros((P,Q),np.float32)
    fp[:M,:N]=1.0*imgin

    # buoc 3:nhan fp với -1^ x+y
    for x in range(0,M):
        for y in range(0,N):
            if (x+y)%2== 1:
                fp[x,y]=-fp[x,y]

    #bước 4
    F=cv2.dft(fp,flags=cv2.DFT_COMPLEX_OUTPUT)
    FR=F[:,:,0]            
    FI=F[:,:,1]
    
    
    # bước 5
    H= CreateNocthFilterRemove(P,Q)
    # buoc 6 nhan G(u,v)=F(u,v)*H(u,v)
    G=cv2.mulSpectrums(F,H,flags=cv2.DFT_ROWS)
    # bươc 7 IDFT
    g =cv2.idft(G,flags=cv2.DFT_SCALE)
    #Bước 8 lấy lại kích thước ban đầu là MxN ,là lấy phần thực,nhân (-1)^(X+Y)
    gR=g[:M,:N,0]
    for x in range(0,M):
        for y in range (0,N):
            if (x+y)%2==1:
                gR[x,y]=-gR[x,y]
    gR=np.clip(gR,0,L-1)
    imgout=gR.astype(np.uint8)           
        


    return imgout
                


