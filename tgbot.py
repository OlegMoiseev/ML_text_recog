import telebot
from datetime import datetime


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

    now = datetime.now()
    filepath = "data_num_recog/tg_messages/" + now.strftime("%d_%m_%Y_%H_%M_%S.png")

    with open(filepath, "wb") as code:
        code.write(file)


bot.polling(none_stop=True, interval=0)
