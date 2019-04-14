 #!/usr/bin/python3
 # -*- coding: utf-8 -*-
import sys

from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from palette import *
import cv2
from functools import partial
import matplotlib.pyplot as plt
from PIL import Image
class myLabel(QLabel):
	clicked = pyqtSignal()
	def mouseReleaseEvent(self, e):
		if e.button() == Qt.LeftButton:
			self.clicked.emit()

class recolorapp(QWidget):
	def __init__(self):
		super(recolorapp, self).__init__()
		self.img = np.array([])

		self.resize(1000,600)
		self.setWindowTitle("Palette-based Recoloring")
		self.palette = None
		self.imagezone = QLabel(self)
		self.imagezone.setScaledContents(True)
		self.imagezone.setFixedSize(600,420)
		self.imagezone.move(80,50)


		#self.btn = QPushButton(self)
		#self.btn.setText("getColor")
		#self.btn.clicked.connect(self.editPalette)
		self.pal_btn = []

		self.img = cv2.cvtColor(cv2.imread("img/tower.png"),cv2.COLOR_BGR2RGB)
		self.orig_img = self.img.copy()
		begin = time.time()
		self.palette = Palette(np.array(self.img)[:,:,0:3])
		end = time.time()
		print("Prepare for palette cost:{}s".format(end-begin))
		#print("palette:",self.palette.centers())
		self.btn = QPushButton(self)
		self.btn.setText("选择图片")
		self.btn.move(80, 10)
		self.btn.clicked.connect(self.openFile)


		self.imagezone2 = QLabel(self)
		self.imagezone2.setScaledContents(True)
		self.imagezone2.setFixedSize(200,140)
		self.imagezone2.move(725,50)
		self.imagezone2.setPixmap(self.cvimg2Pixmap(self.orig_img))
#cv2.cvtColor(img,cv2.COLOR_BGR2RGB,img)
		self.init_palette()
		#btn.clicked.connect(self.dealimage)
		origin_img = np.array(self.img[:,:,0:3],dtype=np.uint8)
		#plt.imshow(origin_img)
		#plt.show()
		self.imagezone.setPixmap(self.cvimg2Pixmap(origin_img))
		#self.dealimage()
	def init_palette(self):
		if self.palette is None:
			return
		tmp = self.palette.centers()
		pnums = tmp.shape[0]

		#palet = tmp
		#blocksize = 50
		#testimg = np.ones((blocksize,blocksize*6,3),dtype=np.uint8)

		#for i in range(6):
		#	testimg[:,(i)*blocksize:(i+1)*blocksize] = palet[i]
		#testimg2 = cv2.cvtColor(testimg,cv2.COLOR_BGR2RGB)
		#cv2.imshow('dddd',testimg)
		#cv2.imshow("cccc",testimg2)
		#cv2.imwrite("test.png",testimg2)
		#print("tmp=",tmp)
		btnsize = 50
		pad = 30
		gap = (self.imagezone.width() - pad*2-pnums*btnsize)/(pnums-1)
		for i in range(0,pnums):
			qcol = QColor(tmp[i][0],tmp[i][1],tmp[i][2])
			btn = myLabel(self)
			btn.setAutoFillBackground(True)
			qp = QPalette()
			qp.setColor(QPalette.Background,qcol)
			btn.setPalette(qp)
			btn.setFixedSize(40,40)
			btn.move(80+pad+i*(btnsize+gap),self.imagezone.height()+self.imagezone.y()+pad)
			btn.clicked.connect(partial(self.editPalette,i))
			self.pal_btn.append(btn)


	def openFile(self):
		filename = QFileDialog.getOpenFileName(self,"Open File","/home/roro/palette_based/img")
		print(filename)
		if filename[0]:
			self.img = cv2.cvtColor(cv2.imread(filename[0]),cv2.COLOR_BGR2RGB)
			self.img = np.array(self.img[:,:,0:3],dtype=np.uint8)
			self.palette = Palette(np.array(self.img)[:,:,0:3])
			self.updatePaletteBtn()

			self.imagezone.setPixmap(self.cvimg2Pixmap(self.img))
			self.orig_img = self.img
			self.imagezone2.setPixmap(self.cvimg2Pixmap(self.orig_img))
	def editPalette(self,idx = 0):
		qcd = QColorDialog(parent=self)
		#qcd.setStandardColor()
		#qcd.setOption(QColorDialog.DontUseNativeDialog,on=True)
		#qcd.setOption(QColorDialog.NoButtons, on=True)
		
		#QColorDialog.ColorDialogOption =  QColorDialog.DontUseNativeDialog
		#col = qcd.getColor()
		qcol = qcd.getColor()
		if qcol.isValid():
			self.updatePalette(idx,qcol)
			self.updatePaletteBtn()
			print(qcol)

	def rgb2Qcolor(self,rgb):
		return QColor(rgb[0],rgb[1],rgb[2])
	def qcolor2Rgb(self,qcol):
		return np.array([qcol.red(),qcol.green(),qcol.blue()])
	def updatePalette(self,idx,qcol):
		new_img = self.palette.update(idx,self.qcolor2Rgb(qcol))
		self.imagezone.setPixmap(self.cvimg2Pixmap(new_img))
	def updatePaletteBtn(self):
		new_p = self.palette.centers()
		for i in range(0,self.palette.K):
			qp = QPalette()
			qp.setColor(QPalette.Background,self.rgb2Qcolor(new_p[i]))
			self.pal_btn[i].setPalette(qp)
		return

	def cvimg2Pixmap(self,img):	
		height, width, bytesPerComponent = img.shape
		qimg = QImage(img.data, width, height, width*3,QImage.Format_RGB888)
		return QPixmap.fromImage(qimg)
	def dealimage(self):
		filename = "tower.png"		
		img = cv2.imread(filename)
		self.showimg(img)

	def showimg(self,img):
		self.imagezone.setPixmap(self.cvimg2Pixmap(img))

def func():
	filename = "tower.png"
	#img = cv2.imread(filename)
	img = Image.open(filename)
	r, g, b, a = img.split()
	#r,g,b = normalize(r),normalize(g),normalize(b)
	#img2 = np.array(list(zip(r,g,b)))
	bin_num = 16

	count = np.zeros((bin_num,bin_num,bin_num),np.float)

	#for v1,v2,v3 in img2:
		




def test():
	
	return 




def testfunc():
	time1 = time.time()
	a = 1.2*3.4+0.5
	time2 = time.time()
	#print("computing..={}",time2-time1)
	qcol = QColor(252,233,79)
	print("Color({},{},{}) picked".format(qcol.red(),qcol.green(), qcol.blue()))
	lab = rgb2lab(np.array([qcol.red(),qcol.green(),qcol.blue()]))
	print("LAB({},{},{}) picked".format(lab[0],lab[1],lab[2]))
	#LAB(95.41246850643135,-10.06887936766998,46.221978157673306) picked

	img = cv2.cvtColor(cv2.imread("tower.png"),cv2.COLOR_BGR2RGB)
	palette = Palette(np.array(img)[:,:,0:3])
	centers_rgb = palette.centers()
	centers_lab = rgb2lab4arr(centers_rgb)
	#palette.showCenters()
	#print("centers_rgb:{} \n centers_lab:{}".format(centers_rgb,centers_lab))
	new_img = palette.update(3,np.array([qcol.red(),qcol.green(),qcol.blue()]))
	#palette.showCenters()
	
	centers_rgb = palette.centers()
	centers_lab = rgb2lab4arr(centers_rgb)
	confirm_rgb = lab2rgb4arr(centers_lab)
	confirm_lab = rgb2lab4arr(confirm_rgb)
	#print("centers_rgb:{} \n centers_lab:{}\nconfirm_rgb:{}\nconfirm_lab:{}".format(centers_rgb,centers_lab,confirm_rgb,confirm_lab))


	plt.imshow(new_img)
	plt.show()

if __name__ == '__main__':
	app = QtWidgets.QApplication(sys.argv);
	myRecolorapp = recolorapp()
	myRecolorapp.show()
	sys.exit(app.exec_())
	#qcol = QColorDialog.getColor()
	

	#testfunc()