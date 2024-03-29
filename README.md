# Detection-of-dangerous-weapons-in-the-hands
Detection of dangerous weapons in the hands using detecto and streamlit

В чём суть репозитория?
------------------------------------
Это финальный проект курса по глубинному обучению Deep Learning School. Моя цель заключалась в том, чтобы обучить уже готовую модель на выбранном мной датасете и сделать Web-demo.

В качестве основного фремйворка для работы с моделью была выбрана библиотека `Detecto`, которая была написана при помощи `PyTorch`, из-за чего если мне не понравится какой-то класс или функция, то я смог бы спокойно их переписать, что я и делал.

В репозитории есть `train.ipynb` файл, который является основным, так как именно в нем была написана модель, датасет, функция измерения метрики MaP, обучена модель и выведены результаты.
Но всё в основном было перенесено в `.py` файлы, так что при запуске `train.py` будет выполнено такое же обучение, но с трекингом в виде MLflow.

Запуск и деплой
------------------------------------
Самый простой вариант запустить модель - использовать подготовленные docker-образы и
файлы `mlflow_tracker.compose.yaml` и `streamlit.compose.yaml`. Первый compose файл запускает MLflow, который следит за тренировкой модели. Второй compose файл отвечает за запуск Streamlit и FastAPI, где первый отвечает за простенький фронтенд, а второй за взаимодействие с самой моделью. Для их запуска необходимо:
1. Запустить docker-контейнер: `docker compose -f mlflow_tracker.compose.yaml up --build -d`
2. Дождаться инициализации MLflow сервера
3. Провести эксперименты и зарегестрировать модель в MLflow
4. Заменить значения enviroment внутри `streamlit.compose.yaml` на ваши и запустить docker-контейнеры: `docker compose -f streamlit.compose.yaml up --build -d`
5. Дождаться инициализации Streamlit и FastAPI

Также можно запустить по-другому: нужно заменить в файлах `mlapi.py`, `web.py` ссылки на `localhost`, ввести в командные строки или терминалы в `PyCharm` команду
* `mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 --port 5000` - запуск MLflow сервера
* `uvicorn mlapi:app --reload` - запуск FastAPI
* `streamlit run web.py` - запуск Streamlit.

Датасет
------------------------------------
Датасет был взят из этого [репозитория](https://github.com/ari-dasci/OD-WeaponDetection) на GitHub. В целом мне понравился датасет тем, что, к примеру, если обучить на нем модель и поставить её на камеры в магазинчике, то можно с большим шансом предотвратить ограбление, если такое имело место быть.

В нем всего 6 классов: нож, банкнота, кредитная карточка, телефон, пистолет и кошелек. Как раз датасет может пригодиться в примере, который я привел выше.

Модель
------------------------------------
В качестве модели был взят Faster R-CNN c ResNet-50 внутри. Сначала был выбор брать какой-нибудь Mobile-Net, чтоб быстрее детекция работала, но результаты такой модели мне не нравились.

Если обратиться к паке `weights/`, то там можно найти 4 файла с весами. Первые три из них как раз относятся к модели полегче, а именно `fasterrcnn_mobilenet_v3_large_fpn` в `PyTorch`, когда как последняя относится уже к тяжелой модельке `fasterrcnn_resnet50_fpn`, которую я в итоге и использовал.


Результаты
------------------------------------
Результаты я покажу на своих фотографиях, где я с ножом и тысячной купюрой.


<p align='center'>
  <img src='photos/before1.jpg' height='512' width='400'/>
  <img src='photos/after1.jpg' height='512' width='400'/>
</p>

<p align='center'>
  <img src='photos/before2.jpg' height='512' width='400'/>
  <img src='photos/after2.jpg' height='512' width='400'/>
</p>

Интересно, что если запускать модель на последней фотографии локально, то будет только бокс "купюры", а удаленно он еще дает бокс с уверенностью 0.06 "ножа".

В итоге я добился Fine-tune модели до метрики `Map = 0.68` и деплоя это в Web, чего и требовалось от проекта.

Источники
------------------------------------
Датасет: <br />
https://github.com/ari-dasci/OD-WeaponDetection <br />

Streamlit:<br />
https://streamlit.io/ <br />
https://habr.com/ru/post/664076/ <br />
https://medium.com/nuances-of-programming/как-развернуть-веб-приложение-streamlit-в-сети-три-простых-способа-3fe4bdbbd0a9  <br />

Detecto: <br />
https://github.com/alankbi/detecto <br />
https://medium.com/pytorch/detecto-build-and-train-object-detection-models-with-pytorch-5f31b68a8109 <br />
