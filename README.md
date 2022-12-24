# ML_text_recog
Project for ML exam #1

Telegram-bot for image processing.
Bot parse input image with numbers and send to user statistics: which numbers were recognised.

Service consists of 3 subsystems:
- Tg-bot to communicate between user and system (upload/download information)
- Image parser (find and cut contours)
- Neural network, teached on MNIST dataset with CUDA.

Scenario of work:
- User send image to bot
- Bot receive it, send to parser
- Gotten POI's send to NN
- NN recognise numbers
- Numbers are counted and send to user in TG

To start this service on your own computer, you need:
- install **requirements.txt**
- add file with your Tg-token to root (without any extensions, txt file)
- start **tgbot.py** file
- enjoy!
