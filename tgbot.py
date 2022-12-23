import telebot
import torchnn
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
        bot.send_message(message.from_user.id, "Hello, user! This Bot designed to recognise and count hand-written numbers. For example, you need to send such image:")
        bot.send_photo(message.from_user.id, photo=open('data_num_recog/example.jpg', 'rb'))
        bot.send_message(message.from_user.id, "After it, you will receive statistics of it")

    else:
        bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")


@bot.message_handler(func=lambda m: True, content_types=['photo'])
def get_broadcast_picture(message):
    file_path = bot.get_file(message.photo[-1].file_id).file_path
    file = bot.download_file(file_path)

    # Convert image to gray and blur it
    src_gray = cv2.imdecode(np.frombuffer(file, np.uint8), -1)

    res = torchnn.recog_nums(src_gray)
    print(res)

    if len(res.most_common()) == 0:
        bot.send_message(message.from_user.id, "There is null image, without any symbols")
    else:
        answer_str = ""
        for value, count in res.most_common():
            answer_str = answer_str + 'Value ' + str(value) + ': ' + str(count) + ' times\n'
        bot.send_message(message.from_user.id, answer_str)


bot.polling(none_stop=True, interval=0)
