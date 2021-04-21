from tkinter import *

from PIL import ImageTk, Image 

from tkinter import filedialog

from tkinter import ttk

import tkinter as tk

import cv2

import numpy as np


import scipy.io as sio
from util.synthetic_util import SyntheticUtil
from util.iou_util import IouUtil
from util.projective_camera import ProjectiveCamera




data = sio.loadmat('data/TranslationData.mat')

# print(data.keys() )
model_points = data['points']
model_line_index = data['line_segment_index']




class MainApplication():
    def __init__(self, master):
        self.master=master
        self.master.title("Synthetic Image Visualization")
        self.master.geometry("1400x920")
        self.pan=Scale(root , from_=-40 , to_=40 , orient= HORIZONTAL)
        self.lan=Scale(root , from_=-40 , to_=40 , orient= HORIZONTAL)

        self.tilt=Scale(root , from_=-20 , to_=-4 , orient= HORIZONTAL)
        self.zoom=Scale(root , from_=1000 , to_=5000 , orient= HORIZONTAL)
        self.Tx=Scale(root , from_=-80 , to_=80 , orient= HORIZONTAL)
        self.Ty=Scale(root , from_=-80 , to_=80 , orient= HORIZONTAL)
        self.Tz=Scale(root , from_= -12 , to_=22 , orient= HORIZONTAL)
        self.roll=Scale(root , from_=-2, to_=2 , orient= HORIZONTAL)


        self.pan.place( x=2, y=2 ,anchor=NW)
        self.panLabel=Label(root, text="PAN")
        self.panLabel.place( x=40, y=40 ,anchor=NW)

        self.lan.place( x=2, y=50 ,anchor=NW)
        self.lanLabel=Label(root, text="LAN")
        self.lanLabel.place( x=40, y=80 ,anchor=NW)


        self.tilt.place( x=122, y=2 ,anchor=NW)
        self.tiltLabel= Label(root, text="Tilt")
        self.tiltLabel.place( x=160, y=40 ,anchor=NW)

        self.zoom.place( x=242, y=2 ,anchor=NW)
        self.zoomLabel=Label(root, text="Zoom")
        self.zoomLabel.place(x=280, y=40 ,anchor=NW)

        self.roll.place( x=362, y=2 ,anchor=NW)
        self.rollLabel=Label(root, text="Roll")
        self.rollLabel.place(x=400, y=40 ,anchor=NW)

        self.Tx.place( x=482, y=2 ,anchor=NW)
        self.TxLabel=Label(root, text="Tx")
        self.TxLabel.place(x=530, y=40 ,anchor=NW)

        self.Ty.place( x=602, y=2 ,anchor=NW)
        self.TyLabel=Label(root, text="Ty")
        self.TyLabel.place(x=650, y=40 ,anchor=NW)

        self.Tz.place( x=722, y=2 ,anchor=NW)
        self.TzLabel=Label(root, text="Tz")
        self.TzLabel.place(x=765, y=40 ,anchor=NW)

        self.b1 = Button(root, text = "Load Image", command= self.loadImageFromFileBrowser, height=2, width= 30, bg="yellow") 
        self.b1.place(x=850, y=10 ,anchor=NW)
        
        self.image = ImageTk.PhotoImage(Image.open('./temp.jpg'))
        self.imagePath="./temp.jpg"
        self.mesh=ImageTk.PhotoImage(Image.open('./temp.jpg'))
        self.ImagePanel=Label( image = self.image)
        self.ImagePanel.place( x=60, y=150 ,anchor=NW)

        self.b2 = Button(root, text = "Project", command= self.Project, height=2, width= 30, bg="green") 
        self.b2.place(x=1120, y=10 ,anchor=NW)

    def loadImageFromFileBrowser(self):
        baseImagePath=filedialog.askopenfilename(initialdir='./', title="Select The Base Image")
        self.image = ImageTk.PhotoImage(Image.open(baseImagePath))
        self.imagePath=baseImagePath
        self.ImagePanel = Label( image = self.image)
        self.ImagePanel.place( x=60, y=150 ,anchor=NW)

    def Project(self):

        self.image = ImageTk.PhotoImage(Image.open(self.imagePath))
        self.ImagePanel = Label( image = self.image)
        self.ImagePanel.place( x=60, y=150 ,anchor=NW)
        
        cc_mean = [[ self.Tx.get(), self.Ty.get(),  self.Tz.get()]]#data['cc_mean'] lef entry control cam position on line horizontally
        cc_std = [[0.0, 0.0 , 0.0]]#data['cc_std']
        cc_min = [[ 45.05679141, -66.0702037 ,  10.13871263]]#data['cc_min']
        cc_max = [[ 60.84563315, -16.74178234,  23.01126126]]#data['cc_max']
        cc_statistics = [cc_mean, cc_std, cc_min, cc_max]

        fl_mean = self.zoom.get() #data['fl_mean']
        fl_std = 0#data['fl_std']
        fl_min = fl_mean#data['fl_min']
        fl_max = fl_mean#data['fl_max']
        fl_statistics = [fl_mean, fl_std, fl_min, fl_max]
        roll_statistics = [0,0,0,0]#[0, 0.2, -1.0, 1.0]

        pan_range = [self.pan.get(), self.pan.get()]#[-35.0, 35.0]
        tilt_range = [self.tilt.get(), self.tilt.get()]#[-15.0, -5.0]
        num_camera = 1

        retrieved_camera_data = SyntheticUtil.generate_ptz_cameras(cc_statistics,
                                             fl_statistics,
                                             roll_statistics,
                                             pan_range, tilt_range,
                                             1280/2.0, 720/2.0,
                                             1)        


        finalImage = SyntheticUtil.camera_to_edge_image(retrieved_camera_data[0], model_points, model_line_index,im_h=720, im_w=1280, line_width=4)

        BasicImage= cv2.imread(self.imagePath)




        
        added_image = cv2.addWeighted(BasicImage,0.5,finalImage,0.1,0)

        cv2.imwrite("combined.png", added_image)
        self.meshPath="combined.png"
        self.mesh = ImageTk.PhotoImage(Image.open(self.meshPath))

        self.ImagePanel = Label( image = self.mesh)
        self.ImagePanel.place( x=60, y=150 ,anchor=NW)












if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()
