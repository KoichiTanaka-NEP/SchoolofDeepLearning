# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import io

from keras.models import model_from_json
from keras.utils import np_utils
from keras_utils import image_to_tensor
from keras import optimizers

from PIL import ImageFile

from flask import Flask, request, abort

from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, ImageMessage

from api_data import CHANNEL_ID, CHANNEL_SECRET, CHANNEL_ACCESS_TOKEN

app = Flask(__name__)

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)


def detect(self, img_file):
    labels = []
    with open('../../models/object_detection_test2/labels.txt') as f:
        data1 = f.read()
        labels = data1.split(' ')
    f_model = '../../models/object_detection_test2'
    model_filename = "vgg16tocifar10_model.json"
    weights_filename = "weights2.hdf5"
    print os.path.join(f_model, model_filename)
    json_string = open(os.path.join(f_model, model_filename)).read()
    print json_string
    model = model_from_json(json_string)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum= 0.9), metrics=['accuracy'])
    model.load_weights(os.path.join(f_model,weights_filename))
    absimgname=os.path.abspath(img_file)
    input_tensor = image_to_tensor(absimgname, 150, 150)
    detection = model.predict(input_tensor)[0]
    a = np.array(detection)
    detect_label = labels[a.argmax(0)]
    return detect_label

@app.route("/", methods=['GET'])
def index():
    return 'Hello World!'

@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'


@handler.add(MessageEvent, message=ImageMessage)
def image_text(event):
    pass


@handler.add(MessageEvent, message=TextMessage)
def message_text(event):
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text="写真を送ると、判別してくれるぴょん(^p^)")
    )



if __name__ == "__main__":
    app.run(debug=False, port=5000, host='0.0.0.0')
