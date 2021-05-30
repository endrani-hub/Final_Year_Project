####################################################################

## Importing the packages ##

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import torch
import torch.nn as nn
import cv2
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from torchvision.utils import save_image

####################################################################



####################################################################

## Initializing the application ##

app = Flask(__name__)

UPLOAD_FOLDER = 'D://Final_Year_Project//Final_Year_Project//uploads'

####################################################################



####################################################################

## Homepage setup ##

@app.route('/')
def homepage():
    return render_template('index.html')


####################################################################



####################################################################

## Gender Prediction ##

## Model Load ##

gender_model = load_model('gender_model.h5')

## Predict Function ##
def prediction(path):
    img = cv2.imread(path)
    img = cv2.resize(img , (150 , 150) , cv2.INTER_AREA)
    img_arr = img_to_array(img)
    img_arr = img_arr / 255.
    pred = gender_model.predict(img_arr.reshape(1 , img_arr.shape[0] , img_arr.shape[1] , img_arr.shape[2] ))
    print(pred)
    if pred > 0.3:
        return 'Woman!' 
    else:
        return 'Man!'


## Gender Page ##

@app.route('/gender' , methods=['GET', 'POST'])
def mask_pred():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            file_loc = os.path.join(UPLOAD_FOLDER , image_file.filename)
            image_file.save(file_loc)
            pred = prediction(file_loc)
            return render_template('gender.html' , pred = pred)        
    return render_template('gender.html' , pred = 'Waiting for some Image!')


####################################################################



####################################################################

## Mask Prediction ##

## Model Load ##

mask_model = load_model('model.h5')

## Predict Function ##
def mask_prediction(path):
    img = cv2.imread(path)
    img = cv2.resize(img , (128 , 128) , cv2.INTER_AREA)
    img_arr = img_to_array(img)
    img_arr = img_arr / 255.
    pred = mask_model.predict(img_arr.reshape(1 , img_arr.shape[0] , img_arr.shape[1] , img_arr.shape[2] ))
    if pred > 0.5:
        return 'With MASK!' 
    else:
        return 'Without MASK!'


@app.route('/mask' , methods = ['GET' , 'POST'])
def mask_detector():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            file_loc = os.path.join(UPLOAD_FOLDER , image_file.filename)
            image_file.save(file_loc)
            pred = mask_prediction(file_loc)
            return render_template('mask.html' , pred = pred)    
    return render_template('mask.html')

####################################################################



####################################################################

## Face Generator ##

## Model Load ##

class Generator(nn.Module):
    def __init__(self , in_channel , out_channel , hidden_channel):
        super().__init__()
        self.in_channel = in_channel
        self.gen_model = nn.Sequential(
        self.gen_block(in_channel , hidden_channel * 16 , stride = 1 , kernel_size = 4 , padding = 0),
        self.gen_block(hidden_channel * 16 , hidden_channel * 8 , stride = 2 , kernel_size = 4 , padding = 1),
        self.gen_block(hidden_channel * 8 , hidden_channel * 4 , stride = 2 , kernel_size = 4 , padding = 1),
        self.gen_block(hidden_channel * 4 , hidden_channel * 2 , stride = 2 , kernel_size = 4 , padding = 1),
        nn.ConvTranspose2d(in_channels = hidden_channel * 2 , 
                           out_channels = out_channel , 
                           kernel_size = 4 , 
                           stride = 2 ,
                           padding = 1),
        nn.Tanh()
        )
        
    def gen_block(self , in_channel , out_channel , stride , kernel_size , padding):
        return nn.Sequential(
        nn.ConvTranspose2d(in_channels = in_channel , 
                           out_channels = out_channel , 
                           kernel_size = kernel_size , 
                           stride = stride ,
                           padding = padding),
        nn.BatchNorm2d(num_features = out_channel),
        nn.ReLU()
        )
    
    def forward(self , noise):
        noise = noise.view(noise.shape[0] , self.in_channel , 1 , 1)
        return self.gen_model(noise)
    

## Generating Noise ##

def gen_noise(batch_size = 16 , noise_dim = 100):
    noise = torch.randn(batch_size , noise_dim)
    return noise


generator_model = torch.load('generator_v2.pth')

@app.route('/face' , methods = ['GET' , 'POST'])
def face_generator(): 
    if request.method == 'POST':
        if len(os.listdir('static/images')) != 0:
            os.remove('static/images/pic.jpg')
        generator_model.eval()
        rand_noise = gen_noise().to('cuda')
        fake_img = generator_model(rand_noise).detach().to('cpu')
        save_image(fake_img , 'static/images/pic.jpg')
        return render_template('face_display.html') 
    return render_template('face.html')

#@app.route('/face_display')
#def face_display():
#    return render_template('face_display.html')




if __name__ == '__main__':
    app.run(debug=True)