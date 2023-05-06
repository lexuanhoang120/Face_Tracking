import datetime

import pyttsx3


class Alert:
    def __init__(self):
        self.engine = pyttsx3.init()
        # voices = engine.getProperty('voices')
        self.engine.setProperty("voice", 'vni')
        self.engine.setProperty("rate", 200)

    def alert(self, text):
        if datetime.datetime.now().hour > 12:
            speech = " Tạm biệt " + str(text)
        else:
            speech = " Xin chào " + str(text)
        self.engine.say(speech)
        self.engine.runAndWait()
        return 0
