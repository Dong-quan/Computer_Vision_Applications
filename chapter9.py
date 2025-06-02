import cv2
import numpy as np
L = 256

def Erosion(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (45,45))
    imgout = cv2.erode(imgin, w)
    return imgout

def Dilation(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    imgout = cv2.dilate(imgin,w)
    return imgout

def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    temp = cv2.erode(imgin, w)
    imgout = imgin - temp
    return imgout

def Contour(imgin):
    #Lưu ý : Contour chỉ dùng cho ảnh nhị phân
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    n = len(contour)
    for i in range (0, n-1):
        x1 = contour[i,0,0]
        y1 = contour[i,0,1]

        x2 = contour[i+1,0,0]
        y2 = contour[i+1,0,1]

        cv2.line(imgout, (x1,y1), (x2, y2), (225,225,0), 2)
    x1 = contour[n-1,0,0]
    y1 = contour[n-1,0,1]

    x2 = contour[0,0,0]
    y2 = contour[0,0,1]

    cv2.line(imgout, (x1,y1), (x2, y2), (225,225,0), 2)
    return imgout

def ConvexHull(imgin):
    #Buoc 1: Tính contour
    #Buoc 2: 
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    hull = cv2.convexHull(contour)
    n = len(hull)
    for i in range (0, n-1):
        x1 = hull[i,0,0]
        y1 = hull[i,0,1]

        x2 = hull[i+1,0,0]
        y2 = hull[i+1,0,1]

        cv2.line(imgout, (x1,y1), (x2, y2), (225,0,225), 2)
    x1 = hull[n-1,0,0]
    y1 = hull[n-1,0,1]

    x2 = hull[0,0,0]
    y2 = hull[0,0,1]

    cv2.line(imgout, (x1,y1), (x2, y2), (225,0,225), 2)
    return imgout

def DefectDetect(imgin):
    #Phai qua 3 buoc
    #buoc1: Contour
    #Buoc2: Tinh bao loi Convex Hull
    #Buoc 3: Tinh DefectDetect

    #Buoc 1 Tinh Contour
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = contours[0]
    #Buoc 2: Tinh bao lồi của contour dạng point
    p = cv2.convexHull(contour, returnPoints= False) #p shape:(18,1) vi tri tren Contour
    n = len(p)
    for i in range (0, n-1):
        vi_tri_1 = p[i, 0]
        vi_tri_2 = p[i+1, 0]

        x1 = contour[vi_tri_1, 0, 0]
        y1 = contour[vi_tri_1, 0, 1]
        x2 = contour[vi_tri_2, 0, 0]
        y2 = contour[vi_tri_2, 0, 1]

        cv2.line(imgout, (x1,y1), (x2, y2), (0,0,225), 2)
    vi_tri_1 = p[n-1, 0]
    vi_tri_2 = p[0, 0]

    x1 = contour[vi_tri_1, 0, 0]
    y1 = contour[vi_tri_1, 0, 1]
    x2 = contour[vi_tri_2, 0, 0]
    y2 = contour[vi_tri_2, 0, 1]

    cv2.line(imgout, (x1,y1), (x2, y2), (0,0,225), 2)    
    #Bước 3: Tính chỗ khuyết
    defects = cv2.convexityDefects(contour, p)
    nguong_do_sau = np.max(defects[:,:,3])//2
    n = len(defects)
    for i in range (0, n):
        do_sau = defects[i,0,3]
        if do_sau > nguong_do_sau:
            vi_tri_khuyet = defects[i,0,2]
            x = contour[vi_tri_khuyet, 0, 0]
            y = contour[vi_tri_khuyet, 0, 1]
            cv2.circle(imgout, (x,y), 5, (0,225,0), -1)
    return imgout

def HoleFill(imgin):
    imgout = cv2.cvtColor(imgin,cv2.COLOR_GRAY2BGR)
    cv2.floodFill(imgout,None,(104,295),(0,0,255))
    return imgout

def ConnectedCompontents(imgin):
    nguong= 200
    _,temp=cv2.threshold(imgin,nguong,L-1,cv2.THRESH_BINARY)
    imgout= cv2.medianBlur(temp,7)
    dem, label=cv2.connectedComponents((imgout),None)
    a = np.zeros(dem,np.int32)
    M,N=label.shape
    for x in range (0,M):
        for y in range (0,N):
            r = label[x,y]
            if r>0:
                a[r]=a[r]+1
    s= ' Co %d thanh phan lien thong'% (dem-1)
    cv2.putText(imgout, s, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255)) 
    for r in range (1, dem):
        s = '%3d %5d' %(r, a[r])
        cv2.putText(imgout, s, (10, (r+1)*15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255)) 
    return imgout

def RemoveSmallRice(imgin):
    # Làm đậm bóng của hạt gạo dùng biến đổi Top_ hat
    #81 là kích thước lơn snhaat của hạt gạo đa bằng (pixed)
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(81,81))
    temp = cv2.morphologyEx(imgin,cv2.MORPH_TOPHAT,w)
    nguong = 120
    _,temp= cv2.threshold(temp,nguong,L-1,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dem, label = cv2.connectedComponents(temp,None)
    a = np.zeros(dem,np.int32)
    M,N = label.shape
    for x in range (0,M):
        for y in range (0,N):
            r = label[x,y]
            if r > 0:
                a[r]=a[r]+1
    max_value= np.max(a)
    imgout =np.zeros((M,N),np.uint8)
    for x in range (0,M):
        for y in range (0,N):
            r = label[x,y]
            if r>0:
                if a[r]> 0.7 *max_value // 2:
                    imgout[x,y]=L-1
    
    return imgout
