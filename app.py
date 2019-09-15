from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import json

import base64
from io import BytesIO

# fastai
from fastai.vision import load_learner
from fastai.vision import open_image
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.model import MBConvBlock
from efficientnet_pytorch.utils import Identity,Conv2dStaticSamePadding,BlockArgs,GlobalParams
import matplotlib.pyplot as plt

#torch
import torch

import numpy as np
import cv2
from PIL import Image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


model_path="dr.pth"
dr_model=load_learner('models/',model_path)
gl_model = torch.load('models/gl.pth')

decode_prediction=["No DR","Mild","Moderate","Severe","Proliferative DR"]

def img_boundary(img_,mask_,kernel):
    disk=mask_[:,:,1]
#    disk=mask_
    disk=disk.astype(np.uint8)
    contours, hierarchy = cv2.findContours(disk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
    cv2.drawContours(img_, contours, -1, (127, 127, 127), 3)
    cnt1=contours[0]
    cnt2=contours[1]
    xdt,ydt = tuple(cnt1[cnt1[:,:,1].argmin()][0])
    xdl,ydl = tuple(cnt1[cnt1[:,:,1].argmax()][0])
    xct,yct = tuple(cnt2[cnt2[:,:,1].argmin()][0])
    xcl,ycl = tuple(cnt2[cnt2[:,:,1].argmax()][0])
    return(img_,xdt,ydt,xdl,ydl,xct,yct,xcl,ycl)


@app.route('/show_image/<filename>')
def show_image(filename):
	return send_from_directory("uploads/",(filename))

def model_predict(img_path, learn):
    img =open_image(img_path)
    out=np.round(np.array(learn.predict(img)[1]))[0]
  
    return out

def get_seg(img_path,model):
	DEVICE='cuda'
	image=cv2.imread(img_path)
	image=cv2.resize(image,(512,512))
	show_img=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image= cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('float32')
	img=image/255
	img= np.moveaxis(img,2,0)
	#img=np.expand_dims(img,axis=0)
	x_tensor = torch.from_numpy(img).to(DEVICE).unsqueeze(0)
	pr_mask=gl_model.predict(x_tensor)
	pr_mask=pr_mask[0].cpu().numpy().round()
	pr_mask=np.moveaxis(pr_mask,0,2)

	kernel = np.ones((1,1),np.uint8)
	img_out,xdt,ydt,xdl,ydl,xct,yct,xcl,ycl=img_boundary(show_img.copy(),pr_mask.copy(),kernel)
	cdr=np.sqrt(np.square(xcl-xct)+np.square(ycl-yct))/np.sqrt(np.square(xdl-xdt)+np.square(ydl-ydt))
	return(img_out,cdr)

@app.route('/', methods=['GET'])
def index():
    # Main page
    name = ""
    return render_template('index.html', image_name=name)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path,dr_model)
        pred_class = decode_prediction[int(preds)]   
        try:
            mask,cdr=get_seg(file_path,gl_model)

            result = str(pred_class)+";cup to disk ratio:"+str(cdr)
        
        except:
            mask=np.zeros((512,512,3))
            result = str(pred_class)+";cup to disk ratio:"+"Disk Area Not Visible"
        plt.imsave('uploads/out1.png',mask)
        # Process your result for human
        

        return result
    return None

@app.route('/out')
def show_out():
    name = 'out1.png'
    return render_template('out.html', image_name=name)


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5002), app)
    http_server.serve_forever()



