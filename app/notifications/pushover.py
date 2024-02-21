import requests
import cv2
from .notification import Notification

class Pushover(Notification):
    def __init__(self, user_token, app_token):
        self.user_token = user_token
        self.app_token = app_token

    @property
    def user_token(self):
        return self._user_token

    @user_token.setter
    def user_token(self, user_token):
        self._user_token = user_token

    @property
    def app_token(self):
        return self._app_token

    @app_token.setter
    def app_token(self, app_token):
        self._app_token = app_token

    @property
    def to(self):
        return self._to

    @to.setter
    def to(self, to):
        self._to = to

    @property
    def message(self):
        return self._message

    @message.setter
    def message(self, message):
        self._message = message

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame):
        self._frame = frame

    def send(self):
        cv2.imwrite('detection.jpg', self._frame)
        r = requests.post(self._to, data = {
          "token": self._app_token,
          "user": self._user_token,
          "message": self._message,
          },
          files = {
            "attachment": ("image.jpg", open("detection.jpg", "rb"), "image/jpeg")
          })

    
