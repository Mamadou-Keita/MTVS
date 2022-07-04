import pandas as pd
import argparse
import cv2
from keras.utils import load_img
import numpy as np
from keras.applications.vgg16 import preprocess_input
from skimage.util import view_as_windows
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import tqdm


def chart_regression(pred, y, name="", sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.savefig(name)
    plt.show()


def main(args):

    test = pd.read_csv(args.input)
    print(test.head())

    model = load_model(args.model)


    y= []
    for pos in range(0, len(test), 1):
        temp = test.iloc[pos:pos + 1] 

        for encimg in tqdm.tqdm(temp.encrypted_image):
            encrypted_image = load_img('./PEID/encimg/'+encimg)
            encrypted_image = np.array(encrypted_image)

            patches = view_as_windows(encrypted_image,(256, 256, 3),(256, 256, 3)).reshape((-1,256,256,3))  

            for i in range(4):
                resized = cv2.resize(patches[i], (224,224), interpolation = cv2.INTER_AREA)
                encrypted_batch = np.expand_dims(resized, axis=0)
                encrypted_batch = preprocess_input(encrypted_batch)

                y.append(model.predict(encrypted_batch))

    
    visual_s, image_s = [], []
    for val in y:
        image_s.append(val[0][0][0])
        visual_s.append(val[1][0][0])

    iq = []
    som, j = 0, 1
    for elt in image_s:
        som += elt
        if j >= 4:
            iq.append(som/4)
            j = 0
            som = 0
        j = j+1

    iq = np.array(iq,)
    iq = iq.reshape((220,))

    vs = []
    somme, j = 0, 1
    for elt in visual_s:
        somme += elt
        if j >= 4:
            vs.append(somme/4)
            j = 0
            somme = 0
        j = j+1

    vs = np.array(vs)
    vs = vs.reshape((220,))

    chart_regression(vs, test.visual_quality.values, name='VisualSecurity.png')
    print('VisualSecurity - Spearmanr',spearmanr(test.visual_quality.values, vs))
    print('VisualSecurity - Pearsonr',pearsonr(test.visual_quality.values, vs))

    chart_regression(iq, test.ground_truth.values,  name='VisualQuality.png')
    print('VisualQuality - Spearmanr',spearmanr(test.ground_truth.values, iq))
    print('VisualQuality - Pearsonr',pearsonr(test.ground_truth.values, iq))

    v = {
        'ground_truth':test.visual_quality.values,
        'predicted_value':vs
    }
    visualQuality = pd.DataFrame(v, columns=['ground_truth','predicted_value'])

    i = {
        'ground_truth':test.ground_truth.values,
        'predicted_value':iq
    }
    imgQuality = pd.DataFrame(i, columns=['ground_truth','predicted_value'])

    visualQuality.to_csv('./MultiTaskBloc1FreezeVisualSecurity.csv', index= False)
    imgQuality.to_csv('./MultiTaskBloc1FreezeVisualQuality.csv', index= False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model Testing...')
    parser.add_argument("--model", "-m", help="path to the model", type=str, default='model.h5')
    parser.add_argument("--input", "-i", help="input csv file", type=str, default='test.csv')
    args = parser.parse_args()

    main(args)