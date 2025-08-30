import subprocess
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivymd.app import MDApp
from kivymd.uix.button import MDIconButton
from kivymd.uix.label import MDLabel
from kivy.animation import Animation

KV = '''
BoxLayout:
    orientation: 'vertical'
    spacing: '10dp'

    MDLabel:
        text: 'Selamat datang di Aplikasi Face Recognition'
        halign: 'center'
        font_style: 'H4'

    BoxLayout:
        orientation: 'vertical'
        size_hint_y: None
        height: dp(72)

        MDIconButton:
            icon: "account-plus"
            pos_hint: {'center_x': 0.5}
            on_release: app.open_realtime_face_registration(self)
            user_font_size: "36sp"
            theme_text_color: "Secondary"
            text: "Face Registration"

        MDIconButton:
            icon: "camera"
            pos_hint: {'center_x': 0.5}
            on_release: app.open_realtime_face_recognition(self)
            user_font_size: "36sp"
            theme_text_color: "Secondary"
            text: "Face Recognition"

        MDIconButton:
            icon: "theme-light-dark" # Icon untuk tombol "Mode Dark"
            pos_hint: {'center_x': 0.5}
            on_release: app.toggle_theme()
            user_font_size: "36sp"
            theme_text_color: "Secondary"
            text: "Mode Dark"
'''

class MainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = 'Light' # Default mode light
        self.theme_cls.primary_palette = 'Teal'
        self.theme_cls.primary_hue = '500'
        return Builder.load_string(KV)

    def animate_button(self, button):
        # Animasi saat tombol ditekan
        animation = Animation(size=(button.width*1.2, button.height*1.2), duration=0.2) + Animation(size=(button.width, button.height), duration=0.2)
        animation.start(button)

    def open_realtime_face_recognition(self, instance):
        try:
            subprocess.Popen(['python', 'realtime_face_recognition.py'])
        except Exception as e:
            print(f"Error: {e}")

    def open_realtime_face_registration(self, instance):
        try:
            subprocess.Popen(['python', 'realtime_face_regist.py'])
        except Exception as e:
            print(f"Error: {e}")

    def toggle_theme(self):
        if self.theme_cls.theme_style == 'Light':
            self.theme_cls.theme_style = 'Dark'
        else:
            self.theme_cls.theme_style = 'Light'

if __name__ == '__main__':
    MainApp().run()
