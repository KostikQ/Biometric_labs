import cv2
import matplotlib.pyplot as plt
import numpy as np


class EYE_DETECTION():
    def __init__(self, path_to_img, path_to_haar_face, path_to_haar_eye):
        self.image = cv2.imread(path_to_img)
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image_gr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
        self.haarcascade_face = cv2.CascadeClassifier(path_to_haar_face)
        self.haarcascade_eye = cv2.CascadeClassifier(path_to_haar_eye)
        self.COLOR_FACE = (255, 255, 0)
        self.COLOR_EYE = (255, 0, 255)
        self.COLOR_IRIS = (255, 0, 0)
        self.COLOR_PUPIL = (0, 255, 0)

    def __call__(self, scaleFactorFace=1.1, minNeighborsFace=20, scaleFactorEyes=1.1, minNeighborsEyes=50,
                 iris_medianBlur_ksize=7, iris_thres=160, iris_thres_maxval=255, iris_thres_type=cv2.THRESH_TRUNC,
                 iris_circles_method=cv2.HOUGH_GRADIENT, iris_circles_dp=1, iris_circles_minDist=100, iris_circles_param1=80,
                 iris_circles_param2=20, iris_circles_minRadius=10, iris_circles_maxRadius=25, pupil_medianBlur_ksize=7, pupil_clip_limit=2.0, 
                 pupil_grid_size=(8,8), pupil_thres=50, pupil_thres_maxval=255, pupil_thres_type=cv2.THRESH_BINARY_INV, 
                 pupil_contour_mode=cv2.RETR_EXTERNAL, pupil_contour_method=cv2.CHAIN_APPROX_SIMPLE):
        self.face_detection(scaleFactorFace, minNeighborsFace)
        self.eye_detection(scaleFactorEyes, minNeighborsEyes)
        self.irises = []
        self.pupils = []
        self.irises_centers = []
        self.pupils_centers = []
        for eye in self.eyes:
            iris, iris_center = self.iris_detection(eye, iris_medianBlur_ksize, iris_thres, iris_thres_maxval, iris_thres_type,
                                                   iris_circles_method, iris_circles_dp, iris_circles_minDist, iris_circles_param1,
                                                   iris_circles_param2, iris_circles_minRadius, iris_circles_maxRadius)
            self.irises.append(iris)
            self.irises_centers.append(iris_center)
            pupil, pupil_center = self.pupil_detection(eye, pupil_medianBlur_ksize, pupil_clip_limit, pupil_grid_size,
                                                    pupil_thres, pupil_thres_maxval, pupil_thres_type,
                                                    pupil_contour_mode, pupil_contour_method)
            self.pupils.append(pupil)
            self.pupils_centers.append(pupil_center)
        
    def face_detection(self, scaleFactor, minNeighbors):
        self.face_rect = self.haarcascade_face.detectMultiScale(self.image_gr, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        self.fx, self.fy, self.fw, self.fh = self.face_rect[0]
        self.face = np.copy(self.image[self.fy:self.fy+self.fh, self.fx:self.fx+self.fw])
        self.face_rgb = cv2.cvtColor(self.face, cv2.COLOR_BGR2RGB)
        self.face_gr = cv2.cvtColor(self.face_rgb, cv2.COLOR_RGB2GRAY)
    
    def eye_detection(self, scaleFactor, minNeighbors):
        self.eyes_rect = self.haarcascade_eye.detectMultiScale(self.face_gr, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        self.eyes = []
        for ex, ey, ew, eh in self.eyes_rect[0:2]:
            self.eyes.append(np.copy(self.face[ey:ey+eh, ex:ex+ew]))
            
    def iris_detection(self, eye, medianBlur_ksize, thres, thres_maxval, thres_type,
                        circles_method, circles_dp, circles_minDist, circles_param1,
                        circles_param2, circles_minRadius, circles_maxRadius):
        
        eye_rgb = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
        eye_gr = cv2.cvtColor(eye_rgb, cv2.COLOR_RGB2GRAY)
        eye_median_filtered = cv2.medianBlur(eye_gr, medianBlur_ksize)
        _, threshold_eye = cv2.threshold(eye_median_filtered, thresh=thres, maxval=thres_maxval, type=thres_type)
        circles = cv2.HoughCircles(threshold_eye, method=circles_method, dp=circles_dp, minDist=circles_minDist, 
                                   param1=circles_param1, param2=circles_param2, minRadius=circles_minRadius, 
                                   maxRadius=circles_maxRadius)
        return circles, circles[0][0][0:2]
        
    def pupil_detection(self, eye, pupil_medianBlur_ksize, clip_limit, grid_size, thres, maxval, thres_type,
                                            contour_mode, contour_method):
        eye_rgb = cv2.cvtColor(eye, cv2.COLOR_BGR2RGB)
        eye_gr = cv2.cvtColor(eye_rgb, cv2.COLOR_RGB2GRAY)
        eye_median_filtered = cv2.medianBlur(eye_gr, ksize=pupil_medianBlur_ksize)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        eye_filtered_clahe = clahe.apply(eye_median_filtered)
        _, threshold_eye = cv2.threshold(eye_filtered_clahe, thresh=thres, maxval=maxval, type=thres_type)
        contours, _ = cv2.findContours(threshold_eye, mode=contour_mode, method=contour_method)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        sum_x = sum([i[0][0] for i in contours[0]])
        sum_y = sum([i[0][1] for i in contours[0]])
        center = (sum_x // len(contours[0]), sum_y // len(contours[0]))
        return contours, center
            
    def draw_detection(self):
        
        plt.axis('off')
        plt.title("Input image")
        plt.imshow(self.image_rgb)
        plt.show()

        plt.axis('off')
        plt.title("Input image gray scale")
        plt.imshow(self.image_gr, cmap='gray')
        plt.show()

        face_detection = np.copy(self.image_rgb)
        face_detection = cv2.rectangle(face_detection, (self.fx, self.fy), (self.fx + self.fw, self.fy + self.fh), self.COLOR_FACE, int((self.fh+self.fw)*0.004 + 1))
        plt.axis('off')
        plt.title("Face detection")
        plt.imshow(face_detection)
        plt.show()
        
        eyes_detection = np.copy(self.face_rgb)
        iris_detection = np.copy(self.face_rgb)
        for ex, ey, ew, eh in self.eyes_rect:
            eyes_detection = cv2.rectangle(eyes_detection, (ex, ey), (ex+ew, ey+eh), self.COLOR_EYE, int((self.fw + self.fh)*0.002 + 1))
        plt.axis('off')
        plt.title("Eyes detection")
        plt.imshow(eyes_detection)
        plt.show()

        pupils = []
        for i in range(2):
            ex, ey, ew, eh = self.eyes_rect[i]
            circle = self.irises[i][0][0]
            iris_detection = cv2.circle(iris_detection, (ex + int(circle[0]), int(ey + circle[1])), int(circle[2]), self.COLOR_IRIS, 2)
            pupils.append(np.array([[[point[0][0] + ex, point[0][1] + ey]] for point in self.pupils[i][0]]))
        pupil_detection = np.copy(self.face_rgb)
        pupil_detection = cv2.drawContours(pupil_detection, pupils, -1, self.COLOR_PUPIL, 2)
        plt.axis('off')
        plt.title("Iris detection")
        plt.imshow(iris_detection)
        plt.show()

        plt.axis('off')
        plt.title("Pupils detection")
        plt.imshow(pupil_detection)
        plt.show()
        

    
    