from PyQt6 import QtWidgets,QtGui
from PyQt6.QtMultimedia import QMediaPlayer,QAudioOutput
from PyQt6.QtCore import Qt,QTimer,QUrl
import sys
import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Insect_Recognition(V1.0.0)') #視窗名稱
        self.setFixedSize(980, 550) #設置窗口的高度和寬度
        self.background = QtGui.QPixmap('C:/Users/llomo/Desktop/Denoise Project/image/background.jpg')
        self.filePath = None #路徑
        self.filepath_denoise = None #濾波圖路徑
        self.img = None #原始圖像
        #self.paintEvent('C:/Users/llomo/Desktop/Denoise Project/image/background.jpg')
        self.ui() #介面
        self.run() #媒體播放
        
    def paintEvent(self,event):
        painter = QtGui.QPainter(self)
        # 繪製背景圖片
        painter.drawPixmap(self.rect(), self.background) #rect() 獲取控件宽高。
        super().paintEvent(event)
        
    def ui(self): 
#######################畫框和按鈕(圖像用途)#################################
        #設定可視區大小(濾波前圖片)
        self.grview_bef = QtWidgets.QGraphicsView(self)
        self.grview_bef.setGeometry(20, 45, 460, 350)  #(x, y, width, height)
            
        #設定可視區大小(濾波後圖片)
        self.grview_aft = QtWidgets.QGraphicsView(self)
        self.grview_aft.setGeometry(500, 45, 460, 350) 
        
        #按鈕(選取圖片)
        self.img_sel = QtWidgets.QPushButton(self)
        self.img_sel.move(20, 410)
        self.img_sel.setText('讀取圖片')
        self.img_sel.setStyleSheet('''
            QPushButton {
                font-size:18px;
                color: #3680ab;
                background: #fffbf8;
                border: 2px solid #d8d8d8;
            }
            QPushButton:hover {
                color: #ffc000;
                background: #fffbf8;
            }
        ''')
        self.img_sel.clicked.connect(self.read_image)

        #按鈕(濾波圖片)
        self.img_F = QtWidgets.QPushButton(self)
        self.img_F.move(120, 410)
        self.img_F.setText('圖片去雜訊')
        self.img_F.setStyleSheet('''
            QPushButton {
                font-size:18px;
                color: #3680ab;
                background: #fffbf8;
                border: 2px solid #d8d8d8;
            }
            QPushButton:hover {
                color: #ffc000;
                background: #fffbf8;
            }
        ''')
        self.img_F.clicked.connect(self.filter_image)
        
        #儲存濾波圖片
        self.img_save = QtWidgets.QPushButton(self)
        self.img_save.move(240,410)
        self.img_save.setText('儲存去噪圖片')
        self.img_save.setStyleSheet('''
            QPushButton {
                font-size:18px;
                color: #3680ab;  
                background: #fffbf8;
                border: 2px solid #d8d8d8;
            }
            QPushButton:hover {
                color: #ffc000;
                background: #fffbf8;
            }
        ''')   
        self.img_save.clicked.connect(self.save_image)
        
        #按鈕(識別圖片)
        self.img_rec = QtWidgets.QPushButton(self)
        self.img_rec.move(380,410)
        self.img_rec.setText('圖片識別')
        self.img_rec.setStyleSheet('''
            QPushButton {
                font-size:18px;
                color: #3680ab;  
                background: #fffbf8;
                border: 2px solid #d8d8d8;
            }
            QPushButton:hover {
                color: #ffc000;
                background: #fffbf8;
            }
        ''')
        self.img_rec.clicked.connect(self.recognition_image)
        
        #音樂播放
        self.music_start = QtWidgets.QPushButton(self)
        self.music_start.setGeometry(20, 480, 60, 25)
        self.music_start.setText('播放')
        self.music_start.setStyleSheet('''
            QPushButton {
                font-size:16px;
                color: #3680ab;
                background: #fffbf8;
                border: 2px solid #d8d8d8;
            }
            QPushButton:hover {
                color: #ffc000;
                background: #fffbf8;
            }
        ''')
        self.music_start.clicked.connect(self.start)
        #音樂暫停
        self.music_pause = QtWidgets.QPushButton(self)
        self.music_pause.setGeometry(80, 480, 60, 25)
        self.music_pause.setText('暫停')
        self.music_pause.setStyleSheet('''
            QPushButton {
                font-size:16px;
                color: #3680ab;
                background: #fffbf8;
                border: 2px solid #d8d8d8;
            }
            QPushButton:hover {
                color: #ffc000;
                background: #fffbf8;
            }
        ''')
        self.music_pause.clicked.connect(self.pause)
        #音樂停止
        self.music_stop = QtWidgets.QPushButton(self)
        self.music_stop.setGeometry(140, 480, 60, 25)
        self.music_stop.setText('停止')
        self.music_stop.setStyleSheet('''
            QPushButton {
                font-size:16px;
                color: #3680ab;
                background: #fffbf8;
                border: 2px solid #d8d8d8;
            }
            QPushButton:hover {
                color: #ffc000;
                background: #fffbf8;
            }
        ''')
        self.music_stop.clicked.connect(self.stop)
        
#######################標籤(說明及內容)#################################
        #畫框(圖片_去雜訊前)
        self.img_denoise_pre = QtWidgets.QLabel(self)
        self.img_denoise_pre.setGeometry(180, 15, 200, 20)
        self.img_denoise_pre.setText('圖片_去雜訊前')
        self.img_denoise_pre.setFont(self.text_style('georgia',15,True))
        #畫框(圖片_去雜訊後)
        self.img_denoise_post = QtWidgets.QLabel(self)
        self.img_denoise_post.setGeometry(660, 15, 200, 20)
        self.img_denoise_post.setText('圖片_去雜訊後')
        self.img_denoise_post.setFont(self.text_style('georgia',15,True))
        #預測(介紹)
        self.pred_intr = QtWidgets.QLabel(self)
        self.pred_intr.setGeometry(500, 405, 400, 30)
        self.pred_intr.setText('識別種類 (蝴蝶、蜻蜓、蚱蜢、瓢蟲、蚊子)')
        self.pred_intr.setFont(self.text_style('georgia',15,True))
        #濾波圖片(是否儲存)
        self.img_save = QtWidgets.QLabel(self)
        self.img_save.setGeometry(500, 445, 200, 30) 
        self.img_save.setFont(self.text_style('georgia',15,True))
        self.img_save.setText('去噪圖片 : 未儲存')
        #預測(內容)
        self.pred_res = QtWidgets.QLabel(self)
        self.pred_res.setGeometry(500, 485, 200, 30) 
        self.pred_res.setFont(self.text_style('georgia',15,True))
        self.pred_res.setText('預測昆蟲類別 : null')
        #播放聲音(時間)
        self.music_t = QtWidgets.QLabel(self)
        self.music_t.setGeometry(210, 480, 90, 23)
        self.music_t.setFont(self.text_style('georgia',12,False))
        self.music_t.setText('0.0 / 0.0')

#######################媒體播放(進度條)#################################
        self.music_slider = QtWidgets.QSlider(self)
        self.music_slider.setOrientation(Qt.Orientation.Horizontal)
        self.music_slider.setGeometry(20, 510, 180, 30)
        self.music_slider.setRange(0, 100)
        self.music_slider.sliderMoved.connect(lambda: self.player.setPosition(round(self.music_slider.value()/1000)))
#######################媒體播放(播放路徑)#################################        
        self.player = QMediaPlayer()        # 設定播放器
        self.path = os.getcwd()             # 取得音樂檔案路徑
        self.qurl = QUrl.fromLocalFile('C:/Users/llomo/Desktop/Denoise Project/music/nature.mp3') # 轉換成 QUrl
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)  # 播放器與音樂輸出器綁定
        self.player.setSource(self.qurl)
        self.player.durationChanged.connect(lambda: self.music_slider.setMaximum(round(self.player.duration()/1000)))
        self.player.play()    
           
    def text_style(self,style,size,bold):
        font = QtGui.QFont()     # 建立文字樣式元件
        font.setFamily(style)    # 設定字體
        font.setPointSize(size)  # 文字大小
        font.setBold(bold)       # 粗體
        return font
    
    def read_image(self): #button
        filePath, filterType = QtWidgets.QFileDialog.getOpenFileNames()  # 選取多個檔案
        #是否有該檔案路徑
        if filePath:
            self.filePath = filePath[0]  # 只取第一個選擇的檔案路徑
            
            # 更新QGraphicsView中的圖片
            scene = self.grview_bef.scene() 
            if scene is not None:
                scene.clear()
                self.pred_res.setText('預測昆蟲類別 : null') #更新
                self.img_save.setText('去噪圖片 : 未儲存') #更新
            else:
                scene = QtWidgets.QGraphicsScene()
                self.grview_bef.setScene(scene) 
                
            #載入圖片(設定大小)
            self.img = QtGui.QPixmap(self.filePath) #场景中加入图片
            self.img = self.img.scaled(455,345) 
            scene.addPixmap(self.img) #显示图片
    
    def save_image(self):
        fileimage_name = os.path.basename(self.filePath)  # basename - example.py
        image_name = os.path.splitext(fileimage_name)[0]  # filename - example
        path = 'C:/Users/llomo/Desktop/Denoise Project/image_denoise/' #儲存圖片路
        self.filepath_denoise = path + image_name + '.PNG'
        self.img.save(self.filepath_denoise) #儲存圖片
        self.img_save.setText('去噪圖片 : 已儲存') 
        
        
    def filter_image(self): #button
    
        #opencv讀取圖片 path不能是中文 & 圖片是rgb opencv是bgr
        self.img = cv2.imread(self.filePath)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        
        # 滤波图像
        self.img = cv2.bilateralFilter(self.img, 9, 75, 75)
        
        # 備份濾波圖像(opencv格式)
        self.img_fliter = self.img
        
        # 更新QGraphicsView中的图像
        scene = self.grview_aft.scene()
        if scene is not None:
            scene.clear()
        else:
            scene = QtWidgets.QGraphicsScene()
            self.grview_aft.setScene(scene)
        
        #opencv轉QImgage
        rows, cols, channels = self.img.shape
        bytesPerLine  = channels * cols
        self.img  = QtGui.QPixmap.fromImage(QtGui.QImage(self.img, cols, rows, bytesPerLine, QtGui.QImage.Format.Format_RGB888))
        
        #顯示圖片(設定大小)
        self.img = self.img.scaled(455,345) 
        scene.addPixmap(self.img) 
        
    def recognition_image(self): #button
        # 載入圖片
        image = Image.open(self.filepath_denoise)
        
        # 指定已保存模型的路徑
        model_path = 'C:/Users/llomo/Desktop/Denoise Project/Filter_GUI/insect_model'

        # 假設模型期望的輸入形狀為 (width, height)，進行調整大小
        image = image.resize((64, 64))
        
        # 將圖片轉換為 numpy 陣列
        image = np.array(image)

        # 將圖片轉換為模型所需的批次形狀，例如 (1, width, height, channels)
        input_data = np.expand_dims(image, axis=0)
        
        # 載入模型
        model = tf.keras.models.load_model(model_path)
        
        # 使用載入的模型進行預測
        predictions = model.predict(input_data)
        
        # 處理預測結果
        predicted_label = np.argmax(predictions)
        
        #輸出結果 蝴蝶、蜻蜓、蚱蜢、瓢蟲、蚊子
        if predicted_label == 0 :
            self.pred_res.setText('預測昆蟲類別 : 蝴蝶')
        elif predicted_label == 1:
            self.pred_res.setText('預測昆蟲類別 : 蜻蜓')
        elif predicted_label == 2:
            self.pred_res.setText('預測昆蟲類別 : 蚱蜢')
        elif predicted_label == 3:
            self.pred_res.setText('預測昆蟲類別 : 瓢蟲')
        else :
            self.pred_res.setText('預測昆蟲類別 : 蚊子')
#######################媒體播放(功能)#################################
    def start(self):
        self.music_start.setDisabled(True)
        self.music_pause.setDisabled(False)
        self.music_stop.setDisabled(False)
        self.player.play()
    def pause(self):
        self.music_start.setDisabled(False)
        self.music_pause.setDisabled(True)
        self.music_stop.setDisabled(False)
        self.player.pause()
    def stop(self):
        self.music_start.setDisabled(False)
        self.music_pause.setDisabled(False)
        self.music_stop.setDisabled(True)
        self.player.stop()
    def playmusic(self):
        progress = round(self.player.position()/1000)  #取的目前播放時間
        self.music_slider.setValue(progress) #設定滑桿位置
        self.music_t.setText(f'{str(progress)} s / {str(round(self.player.duration()/1000))} s')
    def run(self):
        self.timer = QTimer()               # 加入定時器
        self.timer.timeout.connect(self.playmusic)   # 設定定時要執行的 function
        self.timer.start(1000)              # 啟用定時器，設定間隔時間為 500 毫秒
        
        
    
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Form = MyWidget()
    Form.show()
    sys.exit(app.exec())
