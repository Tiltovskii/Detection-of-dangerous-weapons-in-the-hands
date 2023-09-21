import io
import streamlit as st
from PIL import Image
from PIL import ImageDraw, ImageFont
import torch
from config import translate, CLASSES
import requests


def load_image():
    """Создание формы для загрузки изображения"""
    # Форма для загрузки изображения средствами Streamlit
    uploaded_file = st.file_uploader(
        label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        # Получение загруженного изображения
        image_data = uploaded_file.getvalue()
        # Показ загруженного изображения на Web-странице средствами Streamlit
        st.image(image_data)
        # Возврат изображения в формате PIL
        return Image.open(io.BytesIO(image_data)), image_data
    else:
        return None, None


def print_image(img, boxes, conf, labels):
    width, height = img.size
    fort_size = height // 50
    for indx, label in enumerate(labels):
        draw = ImageDraw.Draw(img)
        draw.rectangle([boxes[indx][0], boxes[indx][1], boxes[indx][2], boxes[indx][3]], outline='red', width=3)
        try:
            draw.text((boxes[indx][0] + 5, boxes[indx][1] - 2 * fort_size - 5),
                      f"Label: {translate[label]}\nConf: {conf[indx]:.2f}",
                      font=ImageFont.truetype('arial.ttf', size=fort_size),
                      fill='white')
        except:
            draw.text((boxes[indx][0] + 5, boxes[indx][1] - 2 * fort_size - 5),
                      f"Label: {translate[label]}\nConf: {conf[indx]:.2f}",
                      font=ImageFont.truetype('DejaVuSansMono', size=fort_size),
                      fill='white')

    st.image(img)


def print_image_with_max_conf(img, boxes, conf, labels):
    width, height = img.size
    fort_size = height // 50
    indx = torch.argmax(conf)
    draw = ImageDraw.Draw(img)
    draw.rectangle([boxes[indx][0], boxes[indx][1], boxes[indx][2], boxes[indx][3]], outline='red', width=3)
    try:
        draw.text((boxes[indx][0] + 5, boxes[indx][1] - 2 * fort_size - 5),
                  f"Label: {translate[labels[indx]]}\nConf: {conf[indx]:.2f}",
                  font=ImageFont.truetype('arial.ttf', size=fort_size),
                  fill='white')
    except:
        draw.text((boxes[indx][0] + 5, boxes[indx][1] - 2 * fort_size - 5),
                  f"Label: {translate[labels[indx]]}\nConf: {conf[indx]:.2f}",
                  font=ImageFont.truetype('DejaVuSansMono', size=fort_size),
                  fill='white')

    st.image(img)


def predict(img_data):
    response = requests.post(url='http://localhost:8000/predict',
                             files={'file': img_data,
                                    'type ': 'image/jpeg'},
                             headers={
                                 'accept': 'application/json'}
                             )
    results = response.json()
    return results['boxes'], [CLASSES[label-1] for label in results['labels']], results['scores']


# Выводим заголовок страницы средствами Streamlit
st.title('Детекция предметов в руках на изображении')
# Вызываем функцию создания формы загрузки изображения
img, byte_img = load_image()
result = st.button('Распознать изображение')
# Если кнопка нажата, то запускаем распознавание изображения
if result and img:
    boxes, labels, conf = predict(byte_img)
    print_image(img, boxes, conf, labels)
