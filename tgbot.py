import telebot
from datetime import datetime
import tessract
import cv2
import numpy as np


with open('data_num_recog/token', 'r') as f:
    token = f.read()

bot = telebot.TeleBot(token)


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "Привет":
        bot.send_message(message.from_user.id, "Привет, чем я могу тебе помочь?")
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Напиши привет")
    else:
        bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")


@bot.message_handler(func=lambda m: True, content_types=['photo'])
def get_broadcast_picture(message):
    file_path = bot.get_file(message.photo[-1].file_id).file_path
    file = bot.download_file(file_path)

    # Convert image to gray and blur it
    src_gray = cv2.imdecode(np.frombuffer(file, np.uint8), -1)

    # src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # src_gray = cv2.blur(src_gray, (3, 3))

    now = datetime.now()
    filepath = "data_num_recog/tg_messages/" + now.strftime("%d_%m_%Y_%H_%M_%S.jpg")
    cv2.imwrite(filepath, src_gray)


    thresh = 255  # initial threshold
    tessract.thresh_callback(thresh, src_gray)

    recog_nums()



bot.polling(none_stop=True, interval=0)
