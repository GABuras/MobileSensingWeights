import argparse
import numpy as np
from keras.applications.xception import preprocess_input
import keras.utils as image
# from keras.preprocessing import image
from keras.models import load_model
from PIL import Image

# parser = argparse.ArgumentParser()
# parser.add_argument('model')
# parser.add_argument('classes')
# parser.add_argument('image')
# parser.add_argument('--top_n', type=int, default=10)


def main2(mo, cl, im):

    # create model
    model = load_model(mo)

    # load class names
    classes = []
    with open(cl, 'r') as f:
        classes = list(map(lambda x: x.strip(), f.readlines()))

    # load an input image
    img = image.load_img(im, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # predict
    pred = model.predict(x)[0]
    result = [(classes[i], float(pred[i]) * 100.0) for i in range(len(pred))]
    result.sort(reverse=True, key=lambda x: x[1])
    for i in range(2):
        (class_name, prob) = result[i]
        print("Top %d ====================" % (i + 1))
        print("Class name: %s" % (class_name))
        print("Probability: %.2f%%" % (prob))

    #     c = str(print("Class name: %s" % (class_name)))
    #     p = str(print("Probability: %.2f%%" % (prob)))
    # return c , p


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)