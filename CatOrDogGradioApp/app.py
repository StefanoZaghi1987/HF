__all__ = ['learn_test', 'categories', 'image', 'label', 'examples', 'intf', 'is_cat', 'classify_image']

from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

learn_test = vision_learner(dls_train, resnet18, metrics=error_rate)
learn_test.load('model.pkl')

categories = ('Dog', 'Cat')

def classify_image(img):
    pred,idx,probs = learn_test.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.Image()
label = gr.Label()
examples = ['dog.jpg', 'cat.jpg', 'dunno.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)