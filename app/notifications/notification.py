from abc import ABC, abstractmethod

class Notification(ABC):

    @property
    def to(self):
        pass

    @to.setter
    @abstractmethod
    def to(self, to):
        pass

    @property
    def message(self):
        pass

    @message.setter
    @abstractmethod
    def message(self, message):
        pass

    @abstractmethod
    def send(self):
        pass

