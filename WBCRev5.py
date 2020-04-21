#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Author: David R
#

import wx
import numpy as np
import cv2
import tkinter as tk
import time
import os
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from tkinter import filedialog

# INITIALIZE PATH
saveVerify_path = []
filename = 'path.config'
if '_MEIPASS2' in os.environ:
    filename = os.path.join(os.environ['_MEIPASS2'], filenamen)
for line in open(filename):
    saveVerify_path.append(line.rstrip('\n'))
print("Save path: " + saveVerify_path[0])
print("Verify path: " + saveVerify_path[1])

detailOutput = []

class MainFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # MainFrame.__init__
        kwds["style"] = kwds.get("style", 0) | wx.CAPTION | wx.CLIP_CHILDREN | wx.CLOSE_BOX | wx.MINIMIZE_BOX | wx.SYSTEM_MENU
        wx.Frame.__init__(self, *args, **kwds)
        self.btnSettingFolder = wx.BitmapButton(self, wx.ID_ANY, wx.Bitmap("Resources\\FolderButton2.bmp", wx.BITMAP_TYPE_ANY))
        self.btnAbout = wx.BitmapButton(self, wx.ID_ANY, wx.Bitmap("Resources\\AboutButton2.bmp", wx.BITMAP_TYPE_ANY))
        self.btnHelp = wx.BitmapButton(self, wx.ID_ANY, wx.Bitmap("Resources\\HelpButton2.bmp", wx.BITMAP_TYPE_ANY))
        self.panelIO = wx.Panel(self, wx.ID_ANY)
        self.panelVerify = wx.Panel(self, wx.ID_ANY)
        self.txtCtrlWBC = wx.TextCtrl(self, wx.ID_ANY, "\n", style=wx.TE_CENTRE | wx.TE_READONLY)
        self.btnOpenFile = wx.Button(self, wx.ID_ANY, "Open File", style=wx.BORDER_NONE)
        self.btnDetection = wx.Button(self, wx.ID_ANY, "Detect", style=wx.BORDER_NONE)
        self.btnVerify = wx.Button(self, wx.ID_ANY, "Verify", style=wx.BORDER_NONE)
        self.btnVerify.Disable()
        self.btnSave = wx.Button(self, wx.ID_ANY, "Save", style=wx.BORDER_NONE)
        self.btnSave.Disable()

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_BUTTON, self.btnSettingFolder_pressed, self.btnSettingFolder)
        self.Bind(wx.EVT_BUTTON, self.btnAbout_pressed, self.btnAbout)
        self.Bind(wx.EVT_BUTTON, self.btnHelp_pressed, self.btnHelp)
        self.Bind(wx.EVT_BUTTON, self.btnOpenFile_pressed, self.btnOpenFile)
        self.Bind(wx.EVT_BUTTON, self.btnDetection_pressed, self.btnDetection)
        self.Bind(wx.EVT_BUTTON, self.btnVerify_pressed, self.btnVerify)
        self.Bind(wx.EVT_BUTTON, self.btnSave_pressed, self.btnSave)

        # DEFINE IMG PATH
        self.img_path = ''

    def __set_properties(self):
        # MainFrame.__set_properties
        self.SetTitle("Unstained WBC Detection")
        _icon = wx.NullIcon
        _icon.CopyFromBitmap(wx.Bitmap("Resources\\IconProgram.png", wx.BITMAP_TYPE_ANY))
        self.SetIcon(_icon)
        self.SetBackgroundColour(wx.Colour(204, 50, 50))
        self.btnSettingFolder.SetToolTip("Setting Folder")
        self.btnSettingFolder.SetSize(self.btnSettingFolder.GetBestSize())
        self.btnAbout.SetBackgroundColour(wx.Colour(47, 47, 47))
        self.btnAbout.SetToolTip("About")
        self.btnAbout.SetSize(self.btnAbout.GetBestSize())
        self.btnHelp.SetToolTip("Help")
        self.btnHelp.SetSize(self.btnHelp.GetBestSize())
        self.panelIO.SetMinSize((600, 450))
        self.panelIO.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.panelVerify.SetMinSize((600, 450))
        self.panelVerify.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.txtCtrlWBC.SetMinSize((50, 22))
        self.txtCtrlWBC.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        self.btnOpenFile.SetMinSize((295, 40))
        self.btnOpenFile.SetBackgroundColour(wx.Colour(47, 47, 47))
        self.btnOpenFile.SetForegroundColour(wx.Colour(255, 255, 255))
        self.btnOpenFile.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        self.btnOpenFile.SetToolTip("Input file citra")
        self.btnDetection.SetMinSize((295, 40))
        self.btnDetection.SetBackgroundColour(wx.Colour(47, 47, 47))
        self.btnDetection.SetForegroundColour(wx.Colour(255, 255, 255))
        self.btnDetection.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        self.btnDetection.SetToolTip("Deteksi sel darah putih")
        self.btnVerify.SetMinSize((295, 40))
        self.btnVerify.SetBackgroundColour(wx.Colour(47, 47, 47))
        self.btnVerify.SetForegroundColour(wx.Colour(255, 255, 255))
        self.btnVerify.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        self.btnVerify.SetToolTip("Verifikasi hasil deteksi")
        self.btnSave.SetMinSize((295, 40))
        self.btnSave.SetBackgroundColour(wx.Colour(47, 47, 47))
        self.btnSave.SetForegroundColour(wx.Colour(255, 255, 255))
        self.btnSave.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        self.btnSave.SetToolTip("Simpan hasil deteksi")

    def __do_layout(self):
        # MainFrame.__do_layout
        sizer_24 = wx.BoxSizer(wx.VERTICAL)
        grid_sizer_1 = wx.FlexGridSizer(0, 4, 0, 0)
        sizer_26 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_3 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_4 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_6 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_5 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_4.Add((10, 20), 0, 0, 0)
        sizer_5.Add(self.btnSettingFolder, 0, wx.ALIGN_BOTTOM | wx.LEFT | wx.RIGHT, 5)
        sizer_4.Add(sizer_5, 1, wx.EXPAND, 0)
        bitmap_1 = wx.StaticBitmap(self, wx.ID_ANY, wx.Bitmap("Resources\\LogoProgram.png", wx.BITMAP_TYPE_ANY))
        sizer_4.Add(bitmap_1, 0, wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, 10)
        sizer_6.Add((43, 20), 0, 0, 0)
        sizer_6.Add(self.btnAbout, 0, wx.ALIGN_BOTTOM | wx.LEFT | wx.RIGHT, 5)
        sizer_6.Add(self.btnHelp, 0, wx.ALIGN_BOTTOM, 0)
        sizer_6.Add((10, 20), 0, 0, 0)
        sizer_4.Add(sizer_6, 1, wx.EXPAND, 0)
        sizer_24.Add(sizer_4, 1, wx.EXPAND, 0)
        grid_sizer_1.Add((0, 0), 0, 0, 0)
        grid_sizer_1.Add((20, 10), 0, 0, 0)
        grid_sizer_1.Add((20, 10), 0, 0, 0)
        grid_sizer_1.Add((0, 0), 0, 0, 0)
        grid_sizer_1.Add((0, 0), 0, 0, 0)
        grid_sizer_1.Add((0, 0), 0, 0, 0)
        sizer_3.Add((0, 0), 0, 0, 0)
        sizer_3.Add((0, 0), 0, 0, 0)
        sizer_3.Add((0, 0), 0, 0, 0)
        grid_sizer_1.Add(sizer_3, 1, wx.EXPAND, 0)
        grid_sizer_1.Add((0, 0), 0, 0, 0)
        grid_sizer_1.Add((10, 20), 0, 0, 0)
        grid_sizer_1.Add(self.panelIO, 1, wx.BOTTOM | wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        grid_sizer_1.Add(self.panelVerify, 1, wx.BOTTOM | wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        grid_sizer_1.Add((10, 20), 0, 0, 0)
        grid_sizer_1.Add((0, 0), 0, 0, 0)
        label_4 = wx.StaticText(self, wx.ID_ANY, "Unstained WBC Image  |", style=wx.ALIGN_RIGHT)
        label_4.SetForegroundColour(wx.Colour(255, 255, 255))
        label_4.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        sizer_2.Add(label_4, 0, wx.ALIGN_RIGHT | wx.EXPAND | wx.LEFT, 114)
        label_1 = wx.StaticText(self, wx.ID_ANY, "Detected WBC:", style=wx.ALIGN_RIGHT)
        label_1.SetForegroundColour(wx.Colour(255, 255, 255))
        label_1.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        sizer_2.Add(label_1, 0, wx.ALIGN_RIGHT | wx.LEFT | wx.RIGHT, 5)
        sizer_2.Add(self.txtCtrlWBC, 0, wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, 5)
        grid_sizer_1.Add(sizer_2, 1, wx.EXPAND, 0)
        label_5 = wx.StaticText(self, wx.ID_ANY, "Stained WBC Image", style=wx.ALIGN_CENTER)
        label_5.SetForegroundColour(wx.Colour(255, 255, 255))
        label_5.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        grid_sizer_1.Add(label_5, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        grid_sizer_1.Add((0, 0), 0, 0, 0)
        grid_sizer_1.Add((0, 0), 0, 0, 0)
        sizer_1.Add(self.btnOpenFile, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        sizer_1.Add(self.btnDetection, 0, wx.ALL, 5)
        grid_sizer_1.Add(sizer_1, 1, wx.EXPAND, 0)
        sizer_26.Add(self.btnVerify, 0, wx.ALL, 5)
        sizer_26.Add(self.btnSave, 0, wx.ALL, 5)
        grid_sizer_1.Add(sizer_26, 1, wx.EXPAND, 0)
        grid_sizer_1.Add((0, 0), 0, 0, 0)
        grid_sizer_1.Add((0, 0), 0, 0, 0)
        grid_sizer_1.Add((20, 10), 0, 0, 0)
        grid_sizer_1.Add((20, 10), 0, 0, 0)
        grid_sizer_1.Add((0, 0), 0, 0, 0)
        sizer_24.Add(grid_sizer_1, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_24)
        sizer_24.Fit(self)
        self.Layout()

    def btnSettingFolder_pressed(self, event):  
        # OPEN FRAME SETTING
        try:
            frame = SettingFrame(None, wx.ID_ANY, "")
            frame.Show()
        except:
            event.Skip()

    def btnAbout_pressed(self, event):  
        # OPEN FRAME ABOUT
        try:
            frame = AboutFrame(None, wx.ID_ANY, "")
            frame.Show()
        except:
            event.Skip()

    def btnHelp_pressed(self, event):
        # OPEN FRAME HELP
        try:
            frame = HelpFrame(None, wx.ID_ANY, "")
            frame.Show()
        except:
            event.Skip()

    def btnOpenFile_pressed(self, event):  
        # ----------------------------------------------------------------------
        # -------------------- OPEN AND READ IMAGE FILE ------------------------
        # ----------------------------------------------------------------------
        #open file dialog
        tk.Tk().withdraw()
        self.img_path = filedialog.askopenfilename(title="Select Image",
                                              filetypes=[(".JPG", "*.jpg")])

        if self.img_path:
            self.btnSave.Disable()
            self.btnVerify.Disable()
            self.txtCtrlWBC.SetValue("")
            try:
                # BUAT TAMPILIN KE PANEL
                self.bmp = wx.Image(self.img_path,wx.BITMAP_TYPE_JPEG)
                (w, h) = self.bmp.GetSize() # AMBIL SIZE IMG
                (wp, hp) = self.panelIO.GetSize()
                
                # kondisi resize image
                if h > hp:
                    h = hp
                if w > wp:
                    w = wp
                    
                # SCALE + CONVERT TO BITMAP
                self.bmp = self.bmp.Scale(w,h).ConvertToBitmap()
                
                # TARO KE PANEL
                self.obj = wx.StaticBitmap(self.panelIO, -1, self.bmp,(0,0),(wp, hp))
                self.obj.Refresh()

                # CLEAR VERIFY PANEL (setiap openfile, clear panel verify)
                (wp, hp) = self.panelVerify.GetSize()
                emptyImg = wx.Image(wp, hp, True) # create empty image (black)
                emptyImg.Replace(0,0,0,255,255,255) # replace ke 255 (white)
                self.obj = wx.StaticBitmap(self.panelVerify, -1, emptyImg.ConvertToBitmap(),(0,0),(wp, hp))
                self.obj.Refresh()

                # clear detailOutput
                detailOutput.clear()
            except:
                wx.MessageBox(message="Gagal input citra!",
                                      caption='Load Failed',
                                      style=wx.OK | wx.ICON_ERROR)

    def btnDetection_pressed(self, event):
        def detection(img_path):
            detailOutput.clear()
            try:
                img = cv2.imread(img_path, 1) # read img
                detailOutput.append(img) #DETAIL 0
                img = cv2.resize(img,None,fx=0.32, fy=0.32, interpolation = cv2.INTER_AREA) # resize img
                detectedWBC = img.copy()   # reserve for detected WBC area
##                cv2.imshow('original',img)

                # start timer
                timeExec = time.time()
                
                # ----------------------------------------------------------------------
                # ------------------------- GAMMA CORRECTION ---------------------------
                # ----------------------------------------------------------------------
                def adjust_gamma(image, gamma=1.0):
                    invGamma = 1.0 / gamma
                    table = np.array([((i / 255.0) ** invGamma) * 255
                                     for i in np.arange(0, 256)]).astype("uint8")

                    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

                gamma = 0.4   # ubah gamma value                        
                img = adjust_gamma(img, gamma=gamma)

                # ----------------------------------------------------------------------
                # ----------------------- Extracting Image Saturation n Filtering  -------------------------
                # ----------------------------------------------------------------------
                #-----Converting image to HSV Color model----------------------------------- 
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                #Split HSV
                h, s, v = cv2.split(hsv)

                #CLAHE pada Saturation
                clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(8,8))
                cl = clahe.apply(s)

                #Merge
                limg = cv2.merge((cl,s,v))

                #Convert to RGB
                final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
                detailOutput.append(final.copy()) #DETAIL 1

                # ----------------------------------------------------------------------
                # ----------------------- PRE-PROCESSING IMAGE -------------------------
                # ----------------------------------------------------------------------
                # SMOOTHING AND SHARPENING
                final = cv2.GaussianBlur(final, (5,5), 0)
                detailOutput.append(final) #DETAIL 2
                kernel = np.array([[-1,-1,-1],
                                   [-1,9,-1],
                                   [-1,-1,-1]])
                final = cv2.filter2D(final, -1, kernel)
                detailOutput.append(final.copy()) #DETAIL 3

                # GRAYSCALE, THRESHOLD, MORPHOLOGICAL OPERATION
                gray = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7,7), 0)
                detailOutput.append(gray.copy()) #DETAIL 4
                ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

                # create copy untuk var global detail
                detailOutput.append(thresh.copy()) #DETAIL 5

                # Morph Operation
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
                ##kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
                ##opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                ##close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel2)
                dilate = cv2.dilate(thresh,kernel,iterations = 1)
                tempMorph = dilate.copy()
##                cv2.imshow('Morph', tempMorph)

                # FILLING HOLES
                _,contour,hier = cv2.findContours(tempMorph,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contour:
                    cv2.drawContours(tempMorph,[cnt],0,255,-1)

                detailOutput.append(tempMorph.copy()) #DETAIL 6
##                cv2.imshow('Filling Holes', tempMorph)
##                cv2.waitKey(0)
##                cv2.destroyAllWindows()

                # ----------------------------------------------------------------------
                # ---------------------- WATERSHED SEGMENTATION ------------------------
                # ----------------------------------------------------------------------
                # REMOVE NOISE
                tempMorph = cv2.morphologyEx(tempMorph,cv2.MORPH_OPEN,kernel, iterations = 2)
##                cv2.imshow('Noise Removal', tempMorph)
##                cv2.waitKey(0)
##                cv2.destroyAllWindows()

                # DISTANCE TRANSFORM USING EUCLIDEAN DISTANCE
                D = ndimage.distance_transform_edt(tempMorph)
                localMax = peak_local_max(D, indices=False, min_distance=8, labels=thresh)
                 
                # LABELLING USING CCA AND APPLY WATERSHED
                markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
                labels = watershed(-D, markers, mask=tempMorph)
                print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))

                # LOOPING UNTUK SETIAP LABEL WATERSHED
                for label in np.unique(labels):
                   # jika label 0 = background, skip.
                   if label == 0:
                       continue

                   # selain label 0, maka dianggap area segmen RBC dan dijadikan hitam (blackout area)
                   mask = np.zeros(gray.shape, dtype="uint8")
                   dilate[labels == label] = 0

                detailOutput.append(dilate.copy()) #DETAIL 7
                # output image
##                cv2.imshow("Output", dilate)
##                cv2.waitKey(0)

                # ----------------------------------------------------------------------
                # ------------------------- POST-PROCESSING ----------------------------
                # ----------------------------------------------------------------------
                # CLEANING
                tempClean = dilate.copy()
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
                tempClean = cv2.dilate(tempClean, kernel)
                detailOutput.append(tempClean) #DETAIL 8
                ##tempClean = cv2.morphologyEx(tempClean, cv2.MORPH_OPEN, kernel)
##                cv2.imshow("Clean", tempClean)
##                cv2.waitKey(0)

                # CROP WBC AREA DAN TAMPILKAN PADA LAYAR
                _,contours,hier = cv2.findContours(tempClean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                self.WBC_Count = 0
                self.croppedWBC = []
                self.xywh = []
                for cnt in contours:
                    temp = []
                    area = cv2.contourArea(cnt)
                    cv2.drawContours(img, cnt, -1, 255, 1)
                    x, y, w, h = cv2.boundingRect(cnt)
                    ratio = float(w)/h
                    if area < 4450 or area > 11400:
                      continue
                    if ratio < 0.5 or ratio > 2.3:
                      continue
                    self.croppedWBC.append(detectedWBC[y:y+h,x:x+w])
##                  cv2.imshow("Cropped WBC" + str(WBC_Count), self.croppedWBC[WBC_Count])
##                  cv2.drawContours(detectedWBC, cnt, -1, 255, 1)
                    cv2.rectangle(detectedWBC, (x,y), (x+w,y+h), (255,0,0), 1)
                    cv2.putText(detectedWBC, "Area={}".format(self.WBC_Count+1),(x,y-3),0,0.4,(255,0,0),1)
                    temp.extend([x,y,w,h])
                    self.xywh.append(temp)
                    self.WBC_Count += 1
                print(self.xywh)
                detailOutput.append(detectedWBC) #DETAIL 9
                # end timer
                timeExec = time.time() - timeExec

                #### save ####
##                cv2.imwrite("ValidNonWBC/HasilDeteksi/" + os.path.basename(self.img_path)[:-4] + ".jpg", detectedWBC)
                # TAMPILIN AREA WBC KE PANEL DAN TXT CTRL
                if self.WBC_Count > 0:
                    # SET NILAI txtCtrlWBC (menampilkan jumlah WBC yang terdeteksi)
                    self.txtCtrlWBC.SetValue(str(self.WBC_Count))
                    
                    # MUNCULIN KE PANEL DAN RESIZE
                    (wPanel, hPanel) = self.panelIO.GetSize()
                    detectedWBC = cv2.resize(detectedWBC,(wPanel,hPanel), interpolation = cv2.INTER_AREA)
                    ####### save hasil deteksi ########
                    cv2.imwrite("Detection Output/" + os.path.basename(self.img_path)[:-4] + ".jpg", detectedWBC)
                    ###################################
                    detectedWBC = cv2.cvtColor(detectedWBC, cv2.COLOR_BGR2RGB)
                    wxImg = wx.ImageFromBuffer(detectedWBC.shape[1], detectedWBC.shape[0], detectedWBC)
##                    wxImg = wx.Bitmap.FromBuffer(detectedWBC.shape[1], detectedWBC.shape[0], detectedWBC)
##                    wxImg = wxImg.ConvertToImage()
                    
##                    self.bmp = wx.Image(wxImg, wx.BITMAP_TYPE_ANY)
##                    self.bmp = self.bmp.Scale(wPanel,hPanel).ConvertToBitmap()
                    wxBitmap = wxImg.ConvertToBitmap()
                    
                    # TARO KE PANEL
                    self.obj = wx.StaticBitmap(self.panelIO, -1, wxBitmap,(0,0),(wPanel, hPanel))
                    self.obj.Refresh()
                else:
                    self.txtCtrlWBC.SetValue("0")
                    wx.MessageBox(message="Sel darah putih tidak ditemukan.",
                              caption='WBC not found',
                              style=wx.OK | wx.ICON_INFORMATION)
                       
##                cv2.imshow("Ouput Contours", img)
##                cv2.imshow("OuputWBC Contours", detectedWBC)
##                cv2.waitKey(0)
##                cv2.destroyAllWindows()
            except:
                wx.MessageBox(message="Deteksi gagal!",
                                      caption='Detection Failed',
                                      style=wx.OK | wx.ICON_ERROR) 

            # MEASURE TIME
            img_name = os.path.basename(self.img_path)
            print("---- %s seconds ----" % (timeExec))
            timestr = time.strftime("%d-%m-%Y %H:%M:%S")
            file = open("time.txt", "a")
            file.write("Image file: " + img_name + "\n")
            file.write("Testing date: " + timestr + "\n")
            file.write("Detected WBC: " + str(self.WBC_Count) +"\n")
            file.write("Execution time: " + str(timeExec) + "\n---------------------------------------------\n")
            file.close()

        # ----------------------------------------------------------------------
        # ------------------------- START DETECTION ----------------------------
        # ----------------------------------------------------------------------
        if self.img_path:
            detection(self.img_path)
            self.btnVerify.Enable()
            if self.WBC_Count > 0:
                self.btnSave.Enable()
        else:
            wx.MessageBox(message="File citra tidak ditemukan!",
                                      caption='Masukkan Citra Input',
                                      style=wx.OK | wx.ICON_ERROR)

    def btnVerify_pressed(self, event):
        def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
            dim = None
            (h, w) = image.shape[:2]
            if width is None and height is None:
                return image
            if width is None:
                r = height / float(h)
                dim = (int(w * r), height)
                print("W", dim)
                print(dim[0])
            else:
                r = width / float(w)
                dim = (width, int(h * r))
                print("h", dim)
                print(dim[0])
            resized = cv2.resize(image, dim, interpolation = inter)
            return resized
        
        try:
            # ambil hanya nama img (path paling akhir)
            img_name = os.path.basename(self.img_path)
            verify_path = saveVerify_path[1] + img_name

            # load image untuk kasi rectangle area
            verifyImg = cv2.imread(verify_path)
            for i in range(len(self.xywh)):
                cv2.rectangle(verifyImg, (self.xywh[i][0],self.xywh[i][1]),
                              (self.xywh[i][0]+self.xywh[i][2],
                               self.xywh[i][1]+self.xywh[i][3]), (255,0,0), 1)

            ## save verify image (ada box) ##
            cv2.imwrite("Detection Output/" + os.path.basename(self.img_path)[:-4] + "v.jpg", verifyImg)
            ####################
            verifyImg = cv2.cvtColor(verifyImg, cv2.COLOR_BGR2RGB)
            
            (wp, hp) = self.panelVerify.GetSize()            
            (h, w, _) = verifyImg.shape # AMBIL SIZE IMG
            print (wp,hp, w, h)
            # kondisi resize image
            if h > hp:
                h = hp
            if w > wp:
                w = wp

            # klo pake img tanpa crop
##            verifyImg = cv2.resize(verifyImg, (w,h), interpolation = cv2.INTER_AREA)
            # klo pake img dengan crop
            verifyImg = image_resize(verifyImg, height = hp)
            (h, w, _) = verifyImg.shape
            if w > wp:
                verifyImg = image_resize(verifyImg, width = wp)
            
            wxBuffer = wx.ImageFromBuffer(verifyImg.shape[1], verifyImg.shape[0], verifyImg)
            # SCALE + CONVERT TO BITMAP
            wxBitmap = wxBuffer.ConvertToBitmap()
            # TARO KE PANEL
            self.obj = wx.StaticBitmap(self.panelVerify, -1, wxBitmap,(0,0),(wp, hp))
            self.obj.Refresh()
        except:
            wx.MessageBox(message="Citra untuk verifikasi tidak ditemukan!",
                                      caption='Image Verify Not Found',
                                      style=wx.OK | wx.ICON_ERROR)

    def btnSave_pressed(self, event):  
        try:
            save_path = saveVerify_path[0]
            img_name = os.path.basename(self.img_path) # ambil nama img (path paling belakang)
            for i in range(self.WBC_Count): # save cropped area
                cv2.imwrite(save_path + img_name + "_" + str(i) + ".jpg", self.croppedWBC[i])

            # POP-UP success
            wx.MessageBox(message="Area deteksi berhasil disimpan: " + save_path,
                                  caption='Save Success',
                                  style=wx.OK | wx.ICON_INFORMATION)
        except:
            # POP-UP fail
            wx.MessageBox(message="Gagal menyimpan hasil deteksi!",
                                  caption='Save Failed',
                                  style=wx.OK | wx.ICON_ERROR)

# end of class MainFrame

class AboutFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # AboutFrame.__init__
        kwds["style"] = kwds.get("style", 0) | wx.CAPTION | wx.CLIP_CHILDREN | wx.CLOSE_BOX | wx.SYSTEM_MENU
        wx.Frame.__init__(self, *args, **kwds)
        self.SetSize((700, 360))
        self.AboutOK = wx.Button(self, wx.ID_ANY, "OK")

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_BUTTON, self.btnAboutOK_pressed, self.AboutOK)

    def __set_properties(self):
        # AboutFrame.__set_properties
        self.SetTitle("About Unstained WBC Detection")
        _icon = wx.NullIcon
        _icon.CopyFromBitmap(wx.Bitmap("Resources\\AboutButton2.bmp", wx.BITMAP_TYPE_ANY))
        self.SetIcon(_icon)
        self.SetBackgroundColour(wx.Colour(47, 47, 47))
        self.AboutOK.SetMinSize((200, 26))
        self.AboutOK.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        self.AboutOK.SetToolTip("Back to Home")

    def __do_layout(self):
        # AboutFrame.__do_layout
        sizer_27 = wx.BoxSizer(wx.VERTICAL)
        sizer_28 = wx.BoxSizer(wx.VERTICAL)
        grid_sizer_2 = wx.GridSizer(1, 1, 0, 0)
        label_6 = wx.StaticText(self, wx.ID_ANY, "UNSTAINED WHITE BLOOD CELL DETECTION\nVersion 2.1.0 (201118) Release", style=wx.ALIGN_CENTER)
        label_6.SetBackgroundColour(wx.Colour(47, 47, 47))
        label_6.SetForegroundColour(wx.Colour(0, 127, 255))
        label_6.SetFont(wx.Font(16, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift"))
        sizer_28.Add(label_6, 0, wx.ALIGN_CENTER | wx.ALL | wx.EXPAND, 20)
        label_7 = wx.StaticText(self, wx.ID_ANY, "Program pendeteksian sel darah putih pada citra \npreparat tanpa pewarnaan (unstained).", style=wx.ALIGN_CENTER)
        label_7.SetBackgroundColour(wx.Colour(47, 47, 47))
        label_7.SetForegroundColour(wx.Colour(255, 255, 255))
        label_7.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift"))
        sizer_28.Add(label_7, 0, wx.ALIGN_CENTER | wx.ALL, 5)
        static_line_1 = wx.StaticLine(self, wx.ID_ANY, style=wx.LI_VERTICAL)
        static_line_1.SetMinSize((550, 2))
        sizer_28.Add(static_line_1, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        label_11 = wx.StaticText(self, wx.ID_ANY, "David Reynaldo\nLina", style=wx.ALIGN_CENTER)
        label_11.SetBackgroundColour(wx.Colour(47, 47, 47))
        label_11.SetForegroundColour(wx.Colour(255, 255, 255))
        label_11.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift"))
        grid_sizer_2.Add(label_11, 0, wx.ALIGN_CENTER | wx.EXPAND | wx.LEFT | wx.RIGHT, 30)
        sizer_28.Add(grid_sizer_2, 0, wx.EXPAND, 0)
        label_13 = wx.StaticText(self, wx.ID_ANY, "Fakultas Teknologi Informasi - Teknik Informatika\nUniversitas Tarumanagara (2018)", style=wx.ALIGN_CENTER)
        label_13.SetBackgroundColour(wx.Colour(47, 47, 47))
        label_13.SetForegroundColour(wx.Colour(255, 255, 255))
        label_13.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift"))
        sizer_28.Add(label_13, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        sizer_28.Add(self.AboutOK, 0, wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        sizer_27.Add(sizer_28, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_27)
        self.Layout()

    def btnAboutOK_pressed(self, event):
        self.Close()

# end of class AboutFrame

class HelpFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # HelpFrame.__init__
        kwds["style"] = kwds.get("style", 0) | wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.btnDetail = wx.Button(self, wx.ID_ANY, "Detail Program")
        self.HelpOK = wx.Button(self, wx.ID_ANY, "OK")

        self.__set_properties()
        self.__do_layout()
        self.cekDeteksi()

        self.Bind(wx.EVT_BUTTON, self.btnDetail_pressed, self.btnDetail)
        self.Bind(wx.EVT_BUTTON, self.btnHelpOK_pressed, self.HelpOK)

    def cekDeteksi(self):
        if not detailOutput:
            self.btnDetail.Disable()
        else:
            self.btnDetail.Enable()

    def __set_properties(self):
        # HelpFrame.__set_properties
        self.SetTitle("Help")
        _icon = wx.NullIcon
        _icon.CopyFromBitmap(wx.Bitmap("Resources\\HelpButton2.bmp", wx.BITMAP_TYPE_ANY))
        self.SetIcon(_icon)
        self.SetBackgroundColour(wx.Colour(47, 47, 47))
        self.btnDetail.SetMinSize((200, 26))
        self.btnDetail.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        self.btnDetail.SetToolTip("Detail Program")
        self.HelpOK.SetMinSize((200, 26))
        self.HelpOK.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        self.HelpOK.SetToolTip("Back to Home")

    def __do_layout(self):
        # HelpFrame.__do_layout
        sizer_30 = wx.BoxSizer(wx.VERTICAL)
        lblHelp1 = wx.StaticText(self, wx.ID_ANY, "Petunjuk Penggunaan Program", style=wx.ALIGN_CENTER)
        lblHelp1.SetBackgroundColour(wx.Colour(47, 47, 47))
        lblHelp1.SetForegroundColour(wx.Colour(0, 127, 255))
        lblHelp1.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        sizer_30.Add(lblHelp1, 0, wx.ALIGN_CENTER | wx.ALL | wx.EXPAND, 20)
        static_line_2 = wx.StaticLine(self, wx.ID_ANY)
        sizer_30.Add(static_line_2, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 40)
        lblHelp2 = wx.StaticText(self, wx.ID_ANY, "Informasi tombol:\n- Open File\t: Memilih file citra yang ingin dideteksi\n- Detect\t\t: Proses pendeteksian sel darah putih\n- Verify\t\t: Verifikasi hasil deteksi\n- Save\t\t: Menyimpan citra output\n- About\t\t: Informasi mengenai program\n- Help\t\t: Petunjuk penggunaan program\n- Folder Setting\t: Pengaturan lokasi folder verify/save\n\nTutorial menggunakan program:\n1. Tekan tombol 'Open File' dan pilih file citra preparat sel darah tanpa pewarnaan (.jpg) yang ingin dideteksi.\n2. Kemudian tekan tombol 'Detect', maka area hasil deteksi sel darah putih akan ditampilkan.\n3. Tekan tombol verify untuk mencocokkan apakah area yang terdeteksi benar atau tidak.\n4. Tekan tombol save jika ingin menyimpan citra hasil deteksi (output).\n\nDetail program:", style=wx.ST_NO_AUTORESIZE)
        lblHelp2.SetBackgroundColour(wx.Colour(47, 47, 47))
        lblHelp2.SetForegroundColour(wx.Colour(255, 255, 255))
        lblHelp2.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.NORMAL, 0, "Bahnschrift"))
        sizer_30.Add(lblHelp2, 5, wx.LEFT | wx.RIGHT | wx.TOP, 20)
        sizer_30.Add(self.btnDetail, 0, wx.LEFT | wx.RIGHT, 20)
        lblDetail = wx.StaticText(self, wx.ID_ANY, "*Lakukan deteksi terlebih dahulu untuk melihat tahapan pemrosesan program secara detail", style=wx.ST_NO_AUTORESIZE)
        lblDetail.SetBackgroundColour(wx.Colour(47, 47, 47))
        lblDetail.SetForegroundColour(wx.Colour(255, 255, 255))
        lblDetail.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.NORMAL, 0, "Bahnschrift"))
        sizer_30.Add(lblDetail, 0, wx.BOTTOM | wx.LEFT | wx.RIGHT, 20)
        sizer_30.Add(self.HelpOK, 0, wx.ALIGN_CENTER | wx.BOTTOM | wx.LEFT | wx.RIGHT, 20)
        self.SetSizer(sizer_30)
        sizer_30.Fit(self)
        self.Layout()

    def btnDetail_pressed(self, event): 
        detailFrame = DetailFrame(None, wx.ID_ANY, "")
        detailFrame.Show()

    def btnHelpOK_pressed(self, event):  
        self.Close()

# end of class HelpFrame

class SettingFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # SettingFrame.__init__
        kwds["style"] = kwds.get("style", 0) | wx.CAPTION | wx.CLIP_CHILDREN | wx.CLOSE_BOX | wx.SYSTEM_MENU
        wx.Frame.__init__(self, *args, **kwds)
        self.SetSize((665, 335))
        self.text_ctrl_1 = wx.TextCtrl(self, wx.ID_ANY, saveVerify_path[0])
        self.btnSelectSave = wx.Button(self, wx.ID_ANY, "Select")
        self.text_ctrl_2 = wx.TextCtrl(self, wx.ID_ANY, saveVerify_path[1])
        self.btnSelectVerify = wx.Button(self, wx.ID_ANY, "Select")
        self.btnApply = wx.Button(self, wx.ID_ANY, "Apply")

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_BUTTON, self.btnSelectSave_pressed, self.btnSelectSave)
        self.Bind(wx.EVT_BUTTON, self.btnSelectVerify_pressed, self.btnSelectVerify)
        self.Bind(wx.EVT_BUTTON, self.btnApply_pressed, self.btnApply)

    def __set_properties(self):
        # SettingFrame.__set_properties
        self.SetTitle("Folder Setting")
        _icon = wx.NullIcon
        _icon.CopyFromBitmap(wx.Bitmap("Resources\\FolderBtn.bmp", wx.BITMAP_TYPE_ANY))
        self.SetIcon(_icon)
        self.SetBackgroundColour(wx.Colour(47, 47, 47))
        self.text_ctrl_1.SetMinSize((450, 30))
        self.text_ctrl_1.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.NORMAL, 0, "Bahnschrift"))
        self.btnSelectSave.SetMinSize((110, 30))
        self.btnSelectSave.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        self.text_ctrl_2.SetMinSize((450, 30))
        self.text_ctrl_2.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.NORMAL, 0, "Bahnschrift"))
        self.btnSelectVerify.SetMinSize((110, 30))
        self.btnSelectVerify.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        self.btnApply.SetMinSize((250, 30))
        self.btnApply.SetFont(wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))

    def __do_layout(self):
        # SettingFrame.__do_layout
        sizer_7 = wx.BoxSizer(wx.VERTICAL)
        sizer_8 = wx.BoxSizer(wx.VERTICAL)
        sizer_10 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_9 = wx.BoxSizer(wx.HORIZONTAL)
        label_6 = wx.StaticText(self, wx.ID_ANY, "Pengaturan Folder", style=wx.ALIGN_CENTER)
        label_6.SetBackgroundColour(wx.Colour(47, 47, 47))
        label_6.SetForegroundColour(wx.Colour(0, 127, 255))
        label_6.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift"))
        sizer_8.Add(label_6, 0, wx.ALIGN_CENTER | wx.ALL | wx.EXPAND, 20)
        static_line_2 = wx.StaticLine(self, wx.ID_ANY)
        sizer_8.Add(static_line_2, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 40)
        label_7 = wx.StaticText(self, wx.ID_ANY, "\nSave folder path :")
        label_7.SetBackgroundColour(wx.Colour(47, 47, 47))
        label_7.SetForegroundColour(wx.Colour(255, 255, 255))
        label_7.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift"))
        sizer_8.Add(label_7, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 40)
        sizer_9.Add((30, 20), 0, 0, 0)
        sizer_9.Add(self.text_ctrl_1, 0, wx.LEFT | wx.RIGHT, 10)
        sizer_9.Add(self.btnSelectSave, 0, 0, 0)
        sizer_8.Add(sizer_9, 1, wx.EXPAND, 0)
        label_8 = wx.StaticText(self, wx.ID_ANY, "Verify folder path :")
        label_8.SetBackgroundColour(wx.Colour(47, 47, 47))
        label_8.SetForegroundColour(wx.Colour(255, 255, 255))
        label_8.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift"))
        sizer_8.Add(label_8, 0, wx.LEFT | wx.RIGHT, 40)
        sizer_10.Add((30, 20), 0, 0, 0)
        sizer_10.Add(self.text_ctrl_2, 0, wx.LEFT | wx.RIGHT, 10)
        sizer_10.Add(self.btnSelectVerify, 0, 0, 0)
        sizer_8.Add(sizer_10, 1, wx.EXPAND, 0)
        sizer_8.Add(self.btnApply, 0, wx.ALIGN_CENTER | wx.LEFT | wx.RIGHT, 10)
        sizer_8.Add((20, 15), 0, 0, 0)
        sizer_7.Add(sizer_8, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_7)
        self.Layout()

    def btnSelectSave_pressed(self, event):  
        try:
            tk.Tk().withdraw()
            save_dir = filedialog.askdirectory(title="Select Save Folder")
            if save_dir:
                save_dir = save_dir + "/"
                self.text_ctrl_1.SetValue(save_dir)
        except:
            event.Skip()

    def btnSelectVerify_pressed(self, event):  
        try:
            tk.Tk().withdraw()
            verify_dir = filedialog.askdirectory(title="Select Verify Image Folder")
            if verify_dir:
                verify_dir = verify_dir + "/"
                self.text_ctrl_2.SetValue(verify_dir)
        except:
            event.Skip()

    def btnApply_pressed(self, event):
        try:
            # buka dan read data path.config
            with open(filename, 'r') as file:
                data = file.readlines()

            # ubah path sesuai txtctrl
            data[0] = self.text_ctrl_1.GetValue() + "\n"
            data[1] = self.text_ctrl_2.GetValue()

            # overwrite path.config file
            with open(filename, 'w') as file:
                file.writelines(data)

            # update saveVerify_path global
            saveVerify_path[0] = data[0].rstrip()
            saveVerify_path[1] = data[1].rstrip()
        except:
            event.Skip()
            
        # tutup window
        self.Close()

# end of class SettingFrame

class DetailFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        # DetailFrame.__init__
        kwds["style"] = kwds.get("style", 0) | wx.CAPTION | wx.CLIP_CHILDREN | wx.CLOSE_BOX | wx.MAXIMIZE_BOX | wx.MINIMIZE_BOX | wx.SYSTEM_MENU
        wx.Frame.__init__(self, *args, **kwds)
        self.SetSize((1376, 635))
        self.panel_1 = wx.Panel(self, wx.ID_ANY)
        self.panel_2 = wx.Panel(self, wx.ID_ANY)
        self.panel_3 = wx.Panel(self, wx.ID_ANY)
        self.panel_4 = wx.Panel(self, wx.ID_ANY)
        self.panel_5 = wx.Panel(self, wx.ID_ANY)
        self.panel_6 = wx.Panel(self, wx.ID_ANY)
        self.panel_7 = wx.Panel(self, wx.ID_ANY)
        self.panel_8 = wx.Panel(self, wx.ID_ANY)
        self.panel_9 = wx.Panel(self, wx.ID_ANY)
        self.panel_10 = wx.Panel(self, wx.ID_ANY)
        self.panel_13 = wx.Panel(self, wx.ID_ANY)
        self.btnBack = wx.Button(self, wx.ID_ANY, "< Back", style=wx.BORDER_NONE)
        self.panel_14 = wx.Panel(self, wx.ID_ANY)

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_BUTTON, self.btnBack_pressed, self.btnBack)

        # update panel (munculin semua image)
        self.updatePanel()

    def updatePanel(self):
        # convert semua detailoutput ke bitmap
        imgPanel = []
        (wp, hp) = self.panel_1.GetSize()
        for i in range(len(detailOutput)):
            if ((len(detailOutput[i].shape)) == 2): # cek apa img binary/bgr
                temp = cv2.cvtColor(detailOutput[i], cv2.COLOR_GRAY2RGB) # convert
            else:
                temp = cv2.cvtColor(detailOutput[i], cv2.COLOR_BGR2RGB)
            temp = cv2.resize(temp,(wp,hp), interpolation = cv2.INTER_AREA)
            temp = wx.ImageFromBuffer(temp.shape[1], temp.shape[0], temp)
            temp = temp.ConvertToBitmap()
            imgPanel.append(temp) # append list
            
        print(imgPanel)

        self.obj = wx.StaticBitmap(self.panel_1, -1, imgPanel[0],(0,0),(wp, hp))
        self.obj = wx.StaticBitmap(self.panel_2, -1, imgPanel[1],(0,0),(wp, hp))
        self.obj = wx.StaticBitmap(self.panel_3, -1, imgPanel[2],(0,0),(wp, hp))
        self.obj = wx.StaticBitmap(self.panel_4, -1, imgPanel[3],(0,0),(wp, hp))
        self.obj = wx.StaticBitmap(self.panel_5, -1, imgPanel[4],(0,0),(wp, hp))
        self.obj = wx.StaticBitmap(self.panel_6, -1, imgPanel[5],(0,0),(wp, hp))
        self.obj = wx.StaticBitmap(self.panel_7, -1, imgPanel[6],(0,0),(wp, hp))
        self.obj = wx.StaticBitmap(self.panel_8, -1, imgPanel[7],(0,0),(wp, hp))
        self.obj = wx.StaticBitmap(self.panel_9, -1, imgPanel[8],(0,0),(wp, hp))
        self.obj = wx.StaticBitmap(self.panel_10, -1, imgPanel[9],(0,0),(wp, hp))
        self.obj.Refresh()

    def __set_properties(self):
        # DetailFrame.__set_properties
        self.SetTitle("Detail")
        self.SetBackgroundColour(wx.Colour(165, 42, 42))
        self.panel_1.SetMinSize((260, 195))
        self.panel_1.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.panel_2.SetMinSize((260, 195))
        self.panel_2.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.panel_3.SetMinSize((260, 195))
        self.panel_3.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.panel_4.SetMinSize((260, 195))
        self.panel_4.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.panel_5.SetMinSize((260, 195))
        self.panel_5.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.panel_6.SetMinSize((260, 195))
        self.panel_6.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.panel_7.SetMinSize((260, 195))
        self.panel_7.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.panel_8.SetMinSize((260, 195))
        self.panel_8.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.panel_9.SetMinSize((260, 195))
        self.panel_9.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.panel_10.SetMinSize((260, 195))
        self.panel_10.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.btnBack.SetMinSize((600, 30))
        self.btnBack.SetBackgroundColour(wx.Colour(47, 47, 47))
        self.btnBack.SetForegroundColour(wx.Colour(255, 255, 255))
        self.btnBack.SetFont(wx.Font(14, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        self.btnBack.SetToolTip("Back")

    def __do_layout(self):
        # DetailFrame.__do_layout
        sizer_11 = wx.BoxSizer(wx.VERTICAL)
        sizer_12 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_13 = wx.BoxSizer(wx.VERTICAL)
        sizer_15 = wx.BoxSizer(wx.HORIZONTAL)
        grid_sizer_3 = wx.FlexGridSizer(0, 5, 0, 0)
        sizer_12.Add((10, 20), 0, 0, 0)
        sizer_13.Add((20, 10), 0, 0, 0)
        label_6 = wx.StaticText(self, wx.ID_ANY, "Detail Program", style=wx.ALIGN_CENTER)
        label_6.SetBackgroundColour(wx.Colour(165, 42, 42))
        label_6.SetForegroundColour(wx.Colour(255, 255, 255))
        label_6.SetFont(wx.Font(20, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift"))
        sizer_13.Add(label_6, 0, wx.ALIGN_CENTER | wx.ALL | wx.EXPAND, 20)
        grid_sizer_3.Add(self.panel_1, 1, wx.BOTTOM | wx.EXPAND | wx.RIGHT | wx.TOP, 5)
        grid_sizer_3.Add(self.panel_2, 1, wx.ALL | wx.EXPAND, 5)
        grid_sizer_3.Add(self.panel_3, 1, wx.ALL | wx.EXPAND, 5)
        grid_sizer_3.Add(self.panel_4, 1, wx.ALL | wx.EXPAND, 5)
        grid_sizer_3.Add(self.panel_5, 1, wx.BOTTOM | wx.EXPAND | wx.LEFT | wx.TOP, 5)
        label_5 = wx.StaticText(self, wx.ID_ANY, "1. Input Image", style=wx.ALIGN_CENTER)
        label_5.SetForegroundColour(wx.Colour(255, 255, 255))
        label_5.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        grid_sizer_3.Add(label_5, 0, wx.EXPAND, 5)
        label_7 = wx.StaticText(self, wx.ID_ANY, "2. Saturation Value", style=wx.ALIGN_CENTER)
        label_7.SetForegroundColour(wx.Colour(255, 255, 255))
        label_7.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        grid_sizer_3.Add(label_7, 0, wx.EXPAND, 5)
        label_8 = wx.StaticText(self, wx.ID_ANY, "3. Smoothing", style=wx.ALIGN_CENTER)
        label_8.SetForegroundColour(wx.Colour(255, 255, 255))
        label_8.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        grid_sizer_3.Add(label_8, 0, wx.EXPAND, 5)
        label_9 = wx.StaticText(self, wx.ID_ANY, "4. Sharpening", style=wx.ALIGN_CENTER)
        label_9.SetForegroundColour(wx.Colour(255, 255, 255))
        label_9.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        grid_sizer_3.Add(label_9, 0, wx.EXPAND, 5)
        label_10 = wx.StaticText(self, wx.ID_ANY, "5. Grayscale ", style=wx.ALIGN_CENTER)
        label_10.SetForegroundColour(wx.Colour(255, 255, 255))
        label_10.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        grid_sizer_3.Add(label_10, 0, wx.EXPAND, 5)
        grid_sizer_3.Add(self.panel_6, 1, wx.BOTTOM | wx.EXPAND | wx.RIGHT | wx.TOP, 5)
        grid_sizer_3.Add(self.panel_7, 1, wx.ALL | wx.EXPAND, 5)
        grid_sizer_3.Add(self.panel_8, 1, wx.ALL | wx.EXPAND, 5)
        grid_sizer_3.Add(self.panel_9, 1, wx.ALL | wx.EXPAND, 5)
        grid_sizer_3.Add(self.panel_10, 1, wx.BOTTOM | wx.EXPAND | wx.LEFT | wx.TOP, 5)
        label_11 = wx.StaticText(self, wx.ID_ANY, "6. Otsu Threshold", style=wx.ALIGN_CENTER)
        label_11.SetForegroundColour(wx.Colour(255, 255, 255))
        label_11.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        grid_sizer_3.Add(label_11, 0, wx.EXPAND, 5)
        label_12 = wx.StaticText(self, wx.ID_ANY, "7. Morphological Operation", style=wx.ALIGN_CENTER)
        label_12.SetForegroundColour(wx.Colour(255, 255, 255))
        label_12.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        grid_sizer_3.Add(label_12, 0, wx.EXPAND, 5)
        label_13 = wx.StaticText(self, wx.ID_ANY, "8. Watershed", style=wx.ALIGN_CENTER)
        label_13.SetForegroundColour(wx.Colour(255, 255, 255))
        label_13.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        grid_sizer_3.Add(label_13, 0, wx.BOTTOM | wx.EXPAND, 5)
        label_14 = wx.StaticText(self, wx.ID_ANY, "9. Dilation", style=wx.ALIGN_CENTER)
        label_14.SetForegroundColour(wx.Colour(255, 255, 255))
        label_14.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        grid_sizer_3.Add(label_14, 0, wx.EXPAND, 5)
        label_15 = wx.StaticText(self, wx.ID_ANY, "10. Detection", style=wx.ALIGN_CENTER)
        label_15.SetForegroundColour(wx.Colour(255, 255, 255))
        label_15.SetFont(wx.Font(10, wx.SWISS, wx.NORMAL, wx.BOLD, 0, "Bahnschrift SemiBold"))
        grid_sizer_3.Add(label_15, 0, wx.EXPAND, 5)
        sizer_13.Add(grid_sizer_3, 1, wx.EXPAND, 0)
        sizer_15.Add(self.panel_13, 1, 0, 0)
        sizer_15.Add(self.btnBack, 0, wx.TOP, 10)
        sizer_15.Add(self.panel_14, 1, 0, 0)
        sizer_13.Add(sizer_15, 1, wx.EXPAND, 0)
        sizer_13.Add((20, 10), 0, 0, 0)
        sizer_12.Add(sizer_13, 1, wx.EXPAND, 0)
        sizer_12.Add((10, 20), 0, 0, 0)
        sizer_11.Add(sizer_12, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_11)
        self.Layout()

    def btnBack_pressed(self, event):  
        self.Close()

# end of class DetailFrame

class MyApp(wx.App):
    def OnInit(self):
        self.locale = wx.Locale(wx.LANGUAGE_ENGLISH)
        self.frame = MainFrame(None, wx.ID_ANY, "")
        self.SetTopWindow(self.frame)
        self.frame.Show()
        return True

# end of class MyApp

if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
