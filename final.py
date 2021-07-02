import numpy as np
import sys
if "Tkinter" not in sys.modules:
    from tkinter import *
from tkinter import *
from tkinter import filedialog
from tkinter.scrolledtext import ScrolledText
from tkinter.filedialog import askopenfilename
import cv2
import os
import tkinter as tk
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import pickle
import time
import subprocess
from subprocess import Popen
import argparse
import tensorflow as tf
import pytesseract
from csv import writer
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

from subprocess import Popen,PIPE,STDOUT,call





ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str,
	help="path to output video")
ap.add_argument("-y", "--display", type=int, default=1,
	help="whether or not to display output frame to screen")
args = vars(ap.parse_args())



#data = pickle.loads(open('C:\\Python_CV\\Python37\\FACE_RECOGNITION_PROJECTS\\In a Live VideoPicture\\face-recognition-opencv\\encodings.pickle', "rb").read())




class Test():

    def __init__(self):
        
        self.root = Tk()
        self.root.title('NUMBER PLATE DETECTION')
        self.root.geometry('850x567+0+0')
        #self.root.attributes("-fullscreen", True)



        #def select_image():
            #name = askopenfilename(initialdir="C:/Work_CV/Office_Codes/spectrum/",filetypes =(("Image File", "*.jpg"),("All Files","*.*")),title = "Choose a file.")



            #image = cv2.imread(name)

            #print(name)
            #top0 = tk.Toplevel()
            #top0.title("Number Plate Extracted")
            #top0.geometry("300x150+440+200")
            #small = Canvas(top0, bg="white", height=150, width=300)
            #small.pack()

            
            #small.create_text(52,12, text="Plates Database", font=('Times New Roman', '20', 'bold italic'), fill="black", anchor='nw')





            



            #apna = "alpr "+name+ " --c pk -n 1"

            #proc=Popen(apna, stdout=PIPE, shell=True)
            #output=proc.communicate()[0]
            #b = (str(output))


            #plate= (b[29:35])

            
            #plate = (plate.replace("\\",""))

            #print(plate)


            #small.create_text(90,60, text=plate, font=('Times New Roman', '28', 'bold underline'), fill="black", anchor='nw')

            

            
        def about():
            top = tk.Toplevel()
            top.title("Guidance For Execution")
            top.geometry("400x200+180+200")
            t_lbl = tk.Label(top, text="\n\n1. Select the Image and wait for process")
            t_lbl.pack()


            t_lbl2 = tk.Label(top, text="\n\n2. Select the Video and wait for process")
            t_lbl2.pack()


            t_lbl3 = tk.Label(top, text="\n\n3. For Real Time Video Connect Webcam")
            t_lbl3.pack()


        def live_cam():
            cap = cv2.VideoCapture(0)
            classes = ["background", "number plate"]
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            with tf.io.gfile.GFile('num_plate.pb', 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            with tf.compat.v1.Session() as sess:
                sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')
                while (True):
                    _, img = cap.read()
                    rows = img.shape[0]
                    cols = img.shape[1]
                    inp = cv2.resize(img, (220, 220))
                    inp = inp[:, :, [2, 1, 0]]
                    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
                    num_detections = int(out[0][0])
                    for i in range(num_detections):
                        classId = int(out[3][0][i])
                        score = float(out[1][0][i])
                        bbox = [float(v) for v in out[2][0][i]]
                        label = classes[classId]
                        if (score > 0.3):
                            x = bbox[1]*cols
                            y = bbox[0]*rows
                            right = bbox[3]*cols
                            bottom = bbox[2]*rows
                            color = colors[classId]
                            cv2.rectangle(img, (int(x), int(y)), (int(right),
                                  int(bottom)), color, thickness=1)
                            #cv2.rectangle(img, (int(x), int(y)), (int(right),int(y+30)),color, -1)
                            #cv2.putText(img, str(label),(int(x), int(y)),1,2,(255,255,255),2)
                            crop = img[int(y):int(bottom), int(x):int(right)]
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            #mask = np.zeros(gray.shape,np.uint8)
                            #cv2.imshow(' mask',mask)
                            #new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
                            #new_image = cv2.bitwise_and(gray,mask,mask=mask)
                            # cv2.imshow('d',new_image)
                            Cropped = cv2.resize(gray, (300, 100))
                            #edged = cv2.Canny(Cropped, 30, 150)
                            #thresh = cv2.threshold(gray, 225, 200, cv2.THRESH_BINARY_INV)[1]

                            ret, thresh4 = cv2.threshold(
                                Cropped, 120, 255, cv2.THRESH_TOZERO)
                            cv2.imshow('croped', thresh4)
                            text = pytesseract.image_to_string(thresh4, config='--psm 11')
                            print("Detected license plate Number is:", text)
                            csvList = []
                            csvList.append(text)
                            with open('text.csv', 'a', encoding ='utf8') as f_object:
                                writer_object = writer(f_object)
                                writer_object.writerow(csvList)
                                f_object.close()
                    cv2.imshow('Dashboard', img)
                    key = cv2.waitKey(1)
                    if (key == 27):
                        break
            vs.stop()
            cv2.destroyAllWindows()
            
            if writer is not None:
                writer.release()   

        C = Canvas(self.root, bg="blue", height=850, width=567)
        filename = PhotoImage(file = "C:\\Users\\anjaa\\Desktop\\numberplate\\final\\back.png")
        C.create_image(0, 0, image=filename, anchor='nw')

        

        C.create_text(450,50, text="REAL TIME NUMBER PLATE DETECTION", font=('Times New Roman', '18', 'bold'), fill="white")
        C.pack(fill=BOTH, expand=1)




        button1 = Button(C, text = "Capture Video", font=('Times', '14', 'bold italic'),borderwidth=4, highlightthickness=4, highlightcolor="#37d3ff", highlightbackground="#37d3ff",relief=RAISED, command =live_cam)
        button1.configure(width=15, activebackground = "#33B5E5")
        button1.place(x=180, y=180)


        button2 = Button(C, text = "App Details", font=('Times', '14', 'bold italic'), borderwidth=4, highlightthickness=4, highlightcolor="#37d3ff", highlightbackground="#37d3ff",relief=RAISED, command =about)
        button2.configure(width=15, activebackground = "#33B5E5")
        button2.place(x=180, y=260)

        button3 = Button(C, text = "Closing All", font=('Times', '14', 'bold italic'), borderwidth=4, highlightthickness=4, highlightcolor="#37d3ff", highlightbackground="#37d3ff",relief=RAISED, command=self.quit)
        button3.configure(width=15, activebackground = "#33B5E5")
        button3.place(x=180, y=340)





    
##        self.about = Button(C, text="Saved Video Execution", width="30",font=('Helvetica', '12', 'italic'), command=Camera)
##        self.about.pack(padx=5, pady=40)
##
##        self.about_1 = Button(C, text="Image Analysis", width="30", font=('Helvetica', '12', 'italic'), command=select_image)
##        self.about_1.pack(padx=5, pady=45)
##
##        self.about = Button(C, text="Live Webcam Video", width="30",font=('Helvetica', '12', 'italic'), command=live_cam)
##        self.about.pack(padx=5, pady=50)
##
##        self.about = Button(C, text="About Application", width="30", font=('Helvetica', '12', 'italic'),command=about)
##        self.about.pack(padx=5, pady=55)
##
##        good = Button(C, text="Closing the Window", width="30",font=('Helvetica', '12', 'italic'), command=self.quit)
##        good.pack(padx=5, pady=60)

        

        
        


        self.root.mainloop()

    def quit(self):
        self.root.destroy()


app = Test()
