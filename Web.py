import io
import streamlit as st
from Model import NewModel
from PIL import Image
from PIL import ImageDraw, ImageFont
import torch
from pathlib import Path
from Configure import translate
import gdown


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
        return Image.open(io.BytesIO(image_data))
    else:
        return None


@st.cache(allow_output_mutation=True)
def load_model():
    # fasterrcnn_mobilenet_v3_large_fpn
    model = NewModel(['pistol', 'knife', 'billete', 'monedero', 'smartphone', 'tarjeta'],
                     model_name='fasterrcnn_resnet50_fpn')
    try:
        model = model.load('weights/model_weights3.pth',
                           ['pistol', 'knife', 'billete', 'monedero', 'smartphone', 'tarjeta'],
                           model_name='fasterrcnn_resnet50_fpn')

    except:
        url = "https://drive.google.com/drive/folders/11EQMBTiQr1eqCFR-tuYCcpjcBJK4BUKF"
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            gdown.download_folder(url, quiet=True, use_cookies=False)

        model = model.load('Model/model_weights3.pth',
                           ['pistol', 'knife', 'billete', 'monedero', 'smartphone', 'tarjeta'],
                           model_name='fasterrcnn_resnet50_fpn')
    return model


def print_image(img, boxes, conf, labels):
    width, height = img.size
    fort_size = height // 50
    for indx, label in enumerate(labels):
        draw = ImageDraw.Draw(img)
        draw.rectangle([boxes[indx][0], boxes[indx][1], boxes[indx][2], boxes[indx][3]], outline='red', width=3)
        try:
            draw.text((boxes[indx][0] + 5, boxes[indx][1] - 2*fort_size - 5), f"Label: {translate[label]}\nConf: {conf[indx]:.2f}",
                      font=ImageFont.truetype('arial.ttf', size=fort_size),
                      fill='white')
        except:
            font = ImageFont.load_default()
            draw.text((boxes[indx][0] + 5, boxes[indx][1] - 2 * fort_size - 5),
                      f"Label: {translate[label]}\nConf: {conf[indx]:.2f}",
                      font=ImageFont.truetype('dlig', size=fort_size),
                      fill='white')

    st.image(img)


def print_image_with_max_conf(img, boxes, conf, labels):
    indx = torch.argmax(conf)
    draw = ImageDraw.Draw(img)
    draw.rectangle([boxes[indx][0], boxes[indx][1], boxes[indx][2], boxes[indx][3]], outline='red', width=3)
    draw.text((boxes[indx][0] + 5, boxes[indx][1] - 27), f"Label: {translate[labels[indx]]}\nConf: {conf[indx]:.2f}",
              font=ImageFont.truetype("arial"),
              fill='white')
    st.image(img)


model = load_model()
# Выводим заголовок страницы средствами Streamlit
st.title('Детекция предметов в руках на изображении')
# Вызываем функцию создания формы загрузки изображения
img = load_image()

result = st.button('Распознать изображение')
# Если кнопка нажата, то запускаем распознавание изображения
if result and img:
    labels, boxes, conf = model.predict_top(img)
    print_image(img, boxes, conf, labels)
