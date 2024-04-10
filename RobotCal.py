import cv2
import numpy as np
import math
from typing import List,Tuple

def RobotCal(frame: cv2.typing.MatLike) -> Tuple[float, float]:

    pos10 = pos11 = pos12 = pos15 = -1

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50) 
    parameters =  cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    camera_matrix = np.array([[1917.9816403671446,0,926.6434327683447],[0,1911.0427128637868,559.8249210810895],[0,0,1]])
    dist_coeffs =np.array([-0.03626273835527005,-0.7921296942309466,-0.0011375203379303846,0.0010923311887219979,2.0635280952775354])
    parameters.minMarkerDistanceRate = 0.02


    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(frame, dictionary, parameters = parameters)    
    rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, 0.007, camera_matrix, dist_coeffs)
    print(ids)
    print(tvecs)
    for i in range(4):
        if ids[i][0] == 10:
            pos10 = i
        if ids[i][0] == 11:
            pos11 = i
        if ids[i][0] == 12:
            pos12 = i
        if ids[i][0] == 15:
            pos15 = i                                        

    if pos10==-1 or pos12==-1 or pos11==-1 or pos15==-1:
        raise ValueError
    
    x_centerPixel = [] 
    y_centerPixel = []
    
    for i in range(4):
        x_sum = corners[i][0][0][0]+ corners[i][0][1][0]+ corners[i][0][2][0]+ corners[i][0][3][0]
        y_sum = corners[i][0][0][1]+ corners[i][0][1][1]+ corners[i][0][2][1]+ corners[i][0][3][1]            
        x_centerPixel.append(int(x_sum*.25))
        y_centerPixel.append(int(y_sum*.25))

    print(y_centerPixel[pos15])
    print(x_centerPixel[pos15])    
    print(corners)
    print(corners[0][0][1][1])

    L6 = 0.14
    L1 = 0.16
    L2 = 0.16 #np.sqrt((tvecs[0][0][0]-tvecs[1][0][0])**2 +(tvecs[0][0][1]-tvecs[1][0][1])**2)
    h1 = 0.04
    h2 = 0.02
   
    dist_10_12 = np.sqrt((tvecs[pos10][0][0]-tvecs[pos12][0][0])**2+(tvecs[pos10][0][1]-tvecs[pos12][0][1])**2+(tvecs[pos10][0][2]-tvecs[pos12][0][2])**2) 
    dist_10_12_cr = np.sqrt(dist_10_12**2-(h2-h1)**2)
    alfa = np.pi-np.arccos((L1**2+L6**2-dist_10_12_cr**2)/(2*L1*L6))

    dist_11_15 = np.sqrt((tvecs[pos11][0][0]-tvecs[pos15][0][0])**2+(tvecs[pos11][0][1]-tvecs[pos15][0][1])**2+(tvecs[pos11][0][2]-tvecs[pos15][0][2])**2)
    dist_11_15_cr = np.sqrt(dist_11_15**2-h1**2)
    beta = np.arccos((L2**2+L6**2-dist_11_15_cr**2)/(2*L2*L6))

    for rvec, tvec in zip(rvecs, tvecs):
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    print((tvecs[pos11][0][0],tvecs[pos11][0][1]))        
    while True:
        cv2.line(frame,(x_centerPixel[pos11],y_centerPixel[pos11]),(x_centerPixel[pos10],y_centerPixel[pos10]),(255,255,255),2)
        cv2.imshow('Frame', frame)
        # Premi 'q' per uscire dal loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
           
    return [alfa, beta]

