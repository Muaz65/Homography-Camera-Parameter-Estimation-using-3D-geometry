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
model_points = data['points']
model_line_index = data['line_segment_index']


values = {
    "Rot1": 1.781542,
    "Rot2": -0.425225,
    "Rot3": 0.3268195,
    "FocalLength": 2931.8048,
    "Cam-Mid-STD": 26.2208,
    "Cam-Mid-MIN": -37.2913,
    "Cam-Mid-MAX": 53.0794
}


class MainApplication():
    def __init__(self, master):
        self.master = master
        self.master.title("Synthetic Image Visualization")
        self.master.geometry("1400x920")

        # variabels
        self.CurrentChangeValue = 1

        # HeaderFooter
        self.Header = Label(root, text="Parameters", font=("Courier", 35))
        self.Header.place(x=320, y=1, anchor=NW)

        self.Footer = Label(
            root, text=" © Copyright OMNO AI All Rights Reserved", font=("Courier", 10))
        self.Footer.place(x=580, y=900, anchor=NW)

        # Pan Cofiguration
        self.pan_plus = Button(root, text='+', bd='5',
                               command=lambda: self.incrementString(self.pan_midLabel))
        self.pan_plus.pack(side='top')
        self.pan_plus.place(x=30, y=40, anchor=NW)

        self.pan_midLabel = Label(root, text="Rot1= " + str(values['Rot1']))
        self.pan_midLabel.place(x=80, y=45, anchor=NW)

        self.pan_minus = Button(root, text='-', bd='5',
                                command=lambda: self.decrementString(self.pan_midLabel))
        self.pan_minus.pack(side='top')
        self.pan_minus.place(x=240, y=40, anchor=NW)

        # Tilt Cofiguration
        baseX = 300

        self.tilt_plus = Button(root, text='+', bd='5',
                                command=lambda: self.incrementString(self.tilt_midLabel))
        self.tilt_plus.pack(side='top')
        self.tilt_plus.place(x=baseX, y=40, anchor=NW)

        self.tilt_midLabel = Label(root, text="Rot2= " + str(values['Rot2']))
        self.tilt_midLabel.place(x=baseX+50, y=45, anchor=NW)

        self.tilt_minus = Button(root, text='-', bd='5',
                                 command=lambda: self.decrementString(self.tilt_midLabel))
        self.tilt_minus.pack(side='top')
        self.tilt_minus.place(x=baseX + 210, y=40, anchor=NW)

        # Roll Cofiguration
        baseX = 580
        self.roll_plus = Button(root, text='+', bd='5',
                                command=lambda: self.incrementString(self.roll_midLabel))
        self.roll_plus.pack(side='top')
        self.roll_plus.place(x=580, y=40, anchor=NW)

        self.roll_midLabel = Label(root, text="Rot3= " + str(values['Rot3']))
        self.roll_midLabel.place(x=baseX + 50, y=45, anchor=NW)

        self.roll_minus = Button(root, text='-', bd='5',
                                 command=lambda: self.decrementString(self.roll_midLabel))
        self.roll_minus.pack(side='top')
        self.roll_minus.place(x=baseX + 210, y=40, anchor=NW)

        ### Second Row ###

        # Cam-Center STD Cofiguration
        self.camSTD_plus = Button(root, text='+', bd='5',
                                  command=lambda: self.incrementString(self.camSTD_midLabel))
        self.camSTD_plus.pack(side='top')
        self.camSTD_plus.place(x=30, y=70, anchor=NW)

        self.camSTD_midLabel = Label(
            root, text="Cam-Mid-STD= " + str(values['Cam-Mid-STD']))
        self.camSTD_midLabel.place(x=80, y=75, anchor=NW)

        self.camSTD_minus = Button(root, text='-', bd='5',
                                   command=lambda: self.decrementString(self.camSTD_midLabel))
        self.camSTD_minus.pack(side='top')
        self.camSTD_minus.place(x=240, y=70, anchor=NW)

        # Cam-Center Min Cofiguration
        baseX = 300

        self.camMIN_plus = Button(root, text='+', bd='5',
                                  command=lambda: self.incrementString(self.camMIN_midLabel))
        self.camMIN_plus.pack(side='top')
        self.camMIN_plus.place(x=baseX, y=70, anchor=NW)

        self.camMIN_midLabel = Label(
            root, text="Cam-Mid-MIN= " + str(values['Cam-Mid-MIN']))
        self.camMIN_midLabel.place(x=baseX+50, y=75, anchor=NW)

        self.camMIN_minus = Button(root, text='-', bd='5',
                                   command=lambda: self.decrementString(self.camMIN_midLabel))
        self.camMIN_minus.pack(side='top')
        self.camMIN_minus.place(x=baseX + 210, y=70, anchor=NW)

        # Cam-Center Max Cofiguration
        baseX = 580

        self.camMax_plus = Button(root, text='+', bd='5',
                                  command=lambda: self.incrementString(self.camMax_midLabel))
        self.camMax_plus.pack(side='top')
        self.camMax_plus.place(x=baseX, y=70, anchor=NW)

        self.camMax_midLabel = Label(
            root, text="Cam-Mid-MAX= " + str(values['Cam-Mid-MAX']))
        self.camMax_midLabel.place(x=baseX + 50, y=75, anchor=NW)

        self.camMax_minus = Button(root, text='-', bd='5',
                                   command=lambda: self.decrementString(self.camMax_midLabel))
        self.camMax_minus.pack(side='top')
        self.camMax_minus.place(x=baseX + 210, y=70, anchor=NW)

        ### Third Row ###

        # Focal Length Cofiguration
        baseX = 300

        self.focalLength_plus = Button(root, text='+', bd='5',
                                       command=lambda: self.incrementString(self.focalLength_midLabel))
        self.focalLength_plus.pack(side='top')
        self.focalLength_plus.place(x=baseX, y=100, anchor=NW)

        self.focalLength_midLabel = Label(
            root, text="FocalLength= " + str(values['FocalLength']))
        self.focalLength_midLabel.place(x=baseX+50, y=105, anchor=NW)

        self.focalLength_minus = Button(root, text='-', bd='5',
                                        command=lambda: self.decrementString(self.focalLength_midLabel))
        self.focalLength_minus.pack(side='top')
        self.focalLength_minus.place(x=baseX + 210, y=100, anchor=NW)

        #################

        # Load Image Button
        self.b1 = Button(root, text="Load Image",
                         command=self.loadImageFromFileBrowser, height=1, width=18, bg="yellow")
        self.b1.place(x=950, y=70, anchor=NW)

        # Update Value Cofiguration
        self.ChangeVar = tk.StringVar()
        self.ChangeVar.set(self.CurrentChangeValue)
        self.CurrentUpdateValueEntry = Entry(
            root, textvariable=self.ChangeVar)
        self.CurrentUpdateValueEntry.place(x=950, y=40, anchor=NW)

        self.UpdateValue = Button(root, text="Update Change Val",
                                  command=self.UpdateChangeValue, height=1, width=15, bg="yellow")
        self.UpdateValue.place(x=1170, y=40, anchor=NW)

        # Image View
        self.image = ImageTk.PhotoImage(Image.open('./temp.jpg'))
        self.imagePath = "./temp.jpg"
        self.mesh = ImageTk.PhotoImage(Image.open('./temp.jpg'))
        self.ImagePanel = Label(image=self.image)
        self.ImagePanel.place(x=60, y=150, anchor=NW)

        # Quit button Cofiguration
        self.b2 = Button(root, text="Quit",
                         command=root.destroy, height=1, width=15, bg="green")
        self.b2.place(x=1170, y=70, anchor=NW)

    #######Functions########

    def decrementString(self, label):

        currentString = label['text']

        name, number = currentString.split("=")
        number = float(number)
        number -= self.CurrentChangeValue

        values[name] = number

        label['text'] = name + "= " + str(number)
        self.Project()

    def incrementString(self, label):

        currentString = label['text']

        name, number = currentString.split("=")
        number = float(number)
        number += self.CurrentChangeValue

        values[name] = number

        label['text'] = name + "= " + str(number)
        self.Project()

    def UpdateChangeValue(self):
        updatedValue = float(self.ChangeVar.get())
        self.CurrentChangeValue = updatedValue

    def loadImageFromFileBrowser(self):
        baseImagePath = filedialog.askopenfilename(
            initialdir='./', title="Select The Base Image")

        img = cv2.imread(baseImagePath)
        img = cv2.resize(img, (1280, 720))
        cv2.imwrite(baseImagePath, img)

        self.image = ImageTk.PhotoImage(Image.open(baseImagePath))
        self.imagePath = baseImagePath
        self.ImagePanel = Label(image=self.image)
        self.ImagePanel.place(x=60, y=150, anchor=NW)

    def Project(self):

        self.image = ImageTk.PhotoImage(Image.open(self.imagePath))
        self.ImagePanel = Label(image=self.image)
        self.ImagePanel.place(x=60, y=150, anchor=NW)

        retrieved_camera_data = SyntheticUtil.genrateSingleImage(values['Cam-Mid-STD'], values['Cam-Mid-MIN'], values['Cam-Mid-MAX'],
                                                                 values['FocalLength'],
                                                                 values['Rot1'],
                                                                 values['Rot2'], values['Rot3'],
                                                                 1280/2.0, 720/2.0)

        finalImage = SyntheticUtil.camera_to_edge_image(
            retrieved_camera_data[0], model_points, model_line_index, im_h=720, im_w=1280, line_width=4)

        BasicImage = cv2.imread(self.imagePath)

        added_image = cv2.addWeighted(BasicImage, 0.5, finalImage, 0.1, 0)

        cv2.imwrite("combined.png", added_image)
        self.meshPath = "combined.png"
        self.mesh = ImageTk.PhotoImage(Image.open(self.meshPath))

        self.ImagePanel = Label(image=self.mesh)
        self.ImagePanel.place(x=60, y=150, anchor=NW)


if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()
