import cv2
import os
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivymd.app import MDApp
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivy.core.window import Window

faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

path_gambar = "img"
sample = "img/samples/"

if not os.path.exists(path_gambar):
    os.mkdir(path_gambar)

if not os.path.exists(sample):
    os.mkdir(sample)

class FaceRecognitionApp(MDApp):
    def build(self):
        Window.clearcolor = (1, 1, 1, 1)
        self.nama = ""
        layout = BoxLayout(orientation='vertical', spacing=20, padding=20)

        label_nama = Label(text="Nama :", halign="center", font_size=20)
        self.entry_nama = TextInput(font_size=20)
        button = Button(text="Konfirmasi", font_size=20, background_color=(0.42, 0.36, 0.59, 1), color=(1, 1, 1, 1), on_press=self.myClick1)

        layout.add_widget(label_nama)
        layout.add_widget(self.entry_nama)
        layout.add_widget(button)
        return layout

    def Cari(self, *args):
        cam = cv2.VideoCapture(0)
        while cam.isOpened():
            _, frame = cam.read()
            face = faceCascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in face:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 0, 128), 1)
                cv2.imwrite(f'img/samples/{self.nama}.jpg', frame)

            cv2.imshow("Tekan ENTER saat kotak muncul", frame)

            k = cv2.waitKey(1)
            if k == 13:
                break

        cam.release()
        cv2.destroyAllWindows()

    def myClick1(self, instance):
        self.nama = self.entry_nama.text.strip()
        if not self.nama:
            self.dialog = MDDialog(title="Pendataan Wajah", text="Nama tidak boleh kosong.", size_hint=(0.8, 0.2),
                                   buttons=[MDFlatButton(text="Tutup", on_release=self.close_dialog)])
            self.dialog.open()
        else:
            self.dialog = MDDialog(title="Pendataan Wajah", text="Apakah Anda yakin ingin membuka kamera?",
                                   size_hint=(0.8, 0.2),
                                   buttons=[MDFlatButton(text="Batal", on_release=self.close_dialog),
                                            MDFlatButton(text="Iya", on_release=self.Cari)])
            self.dialog.open()

    def close_dialog(self, instance):
        self.dialog.dismiss()

if __name__ == '__main__':
    FaceRecognitionApp().run()