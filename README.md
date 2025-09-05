# deepschool-bentoml-article

### Описание 

Этот репозиторий создан к статье https://blog.deepschool.ru/ Сервинг моделей с BentoML.

Используется модель IDEA-Research/grounding-dino-tiny для обнаружения объектов на изображении, которые соответствуют заданному текстовому запросу.

### Установка
- Установка uv
curl -LsSf https://astral.sh/uv/install.sh | sh

Создаем виртуальное окружение
```
uv venv --python 3.11
```

Активируем виртуальное окружение
```
source .venv/bin/activate
```

- Установка зависимостей из pyproject.toml
```
uv sync 
```

Можно установить конкретную библиотеку так, подробнее https://docs.astral.sh/uv/
```
uv add (название библиотеки)
```

### Запуск базового скрипта с Grounding DINO моделью
```
uv run grounding_dino_demo.py
```
Выведется результат работы модели в консоль. При желании можно сохранить изображение в файл out_demo.jpg раскоментив строку vis.save("out_demo.jpg")

### Запуск для локальной развертки
Запускаем сервис BentoML, сервис будет доступен по адресу http://localhost:3025. По умолчанию сервис будет доступен по адресу http://localhost:3000.
```
bentoml serve --port 3025
```

### Запуск сервиса BentoML в Docker
Для подготовки к сборке образа BentoML необходимо выполнить в терминале команду:
```
bentoml build 
```
После можно создать образ
```
bentoml containerize grounding-dino-service:latest
```
Создать контейнер
```
docker run --rm --gpus '"device=1"' -p 3025:3000 grounding-dino-service:<id_image>
```

### Запуск клиента SDK BentoML
Команда для запуска клиента, выведет результат в логе. Когда ваш сервис будет доступен по адресу http://localhost:3025, то можно запустить клиент.

```
uv run client.py
```
Основной код внутри client.py, синхронный клиент, который будет ждать ответа от сервиса:
```
with bentoml.SyncHTTPClient("http://localhost:3025") as client:
    result = client.detect_image(
        image=img,
        params={
            "detection_prompt": [["a cat", "a remote control"]],
            "box_threshold": 0.25,
            "text_threshold": 0.25,
        },
    )
```
### Запуск клиента CLI BentoML


Первая ручка detect_image
```
curl -X 'POST' \
  'http://localhost:3025/detect_image' \
  -H 'accept: image/*' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@images/original_cat.jpg;type=image/jpeg' \
  -F 'params={
  "detection_prompt": [
    [
      "a cat", "a remote control"
    ]
  ],
  "box_threshold": 0.25,
  "text_threshold": 0.25
};type=application/json'
```

Вторая ручка /render
```
curl -X 'POST' \
  'http://localhost:3025/render' \
  -H 'accept: image/*' \
  -H 'Content-Type: multipart/form-data' \
  -F 'image=@images/original_cat.jpg;type=image/jpeg' \
  -F 'params={
  "detection_prompt": [
    [
      "a cat", "a remote control"
    ]
  ],
  "box_threshold": 0.25,
  "text_threshold": 0.25
};type=application/json' \
  --output images/result_render.jpg
```
