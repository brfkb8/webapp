from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pathlib import Path
import uvicorn, aiohttp, asyncio
import base64, sys, numpy as np, tensorflow as tf

path = Path(__file__).parent
#model_file_url = 'https://github.com/pankymathur/Fine-Grained-Clothing-Classification/blob/master/data/cloth_categories/models/stage-1_sz-150.pth?raw=true'
model_file_url = 'https://github.com/brfkb8/webapp/blob/master/keras_app/app/models/model.h5?raw=true'
model_file_name = 'model'
classes = ['dutch', 'english', 'french', 'german', 'italian', 'japanese', 'javanese', 'korean', 
           'mandarin', 'russian', 'spanish', 'sundanese', 'vietnamese']

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

MODEL_PATH = str(path/'models'/f'{model_file_name}.h5')
IMG_FILE_SRC = str(path/'static'/'saved_image.png')
PREDICTION_FILE_SRC = str(path/'static'/'predictions.txt')

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)


async def setup_model():
    #UNCOMMENT HERE FOR CUSTOM TRAINED MODEL
    await download_file(model_file_url, MODEL_PATH)
    model = tf.keras.models.Sequential()  
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                   input_shape=(128,480,1)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, (1, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (1, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1,3)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(64, (1, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Conv2D(64, (1, 3), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1,3)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.Dense(13, activation='softmax'))

    model.compile(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.0005),
                    loss=tf.keras.losses.categorical_crossentropy, metrics=['acc'])
    print(MODEL_PATH)
    model.load_weights(str(MODEL_PATH))
   #  model = load_model(MODEL_PATH) # Load your Custom trained model
    # model._make_predict_function()
   # model = ResNet50(weights='imagenet') # COMMENT, IF you have Custom trained model
    return model

# Asynchronous Steps
loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    img_bytes = await (data["img"].read())
    bytes = base64.b64decode(img_bytes)
    with open(IMG_FILE_SRC, 'wb') as f: f.write(bytes)
    return model_predict(IMG_FILE_SRC, model)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128, 480),color_mode='grayscale')
    x = np.expand_dims(image.img_to_array(img), axis=0)
    result = classes[np.argmax(model.predict(x,batch=1))]
    with open(PREDICTION_FILE_SRC, 'w') as f: f.write(str(result))
    result_html = str(path/'static'/'result.html')
    return HTMLResponse(result_html.open().read())

@app.route("/")
def form(request):
    index_html = str(path/'static'/'index.html')
    return HTMLResponse(index_html.open().read())

if __name__ == "__main__":
    if "serve" in sys.argv: uvicorn.run(app, host="0.0.0.0", port=8080)
