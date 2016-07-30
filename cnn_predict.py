# -*- coding:utf-8 -*-
import tensorflow as tf
import cnn_setting
import os.path
import sys
from PIL import Image,ImageFilter


def predictint(imvalue):

    init_op = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        saver.restore(sess, "model.ckpt")   
        prediction=tf.argmax(cnn_setting.y,1)
        #print prediction
        print cnn_setting.y.eval(feed_dict={cnn_setting.x: [imvalue],cnn_setting.keep_prob: 1.0}, session=sess)
        return prediction.eval(feed_dict={cnn_setting.x: [imvalue],cnn_setting.keep_prob: 1.0}, session=sess)

def imageprepare(argv):

    imOrigin = Image.open(argv)
    imOrigin.load() 

    im = Image.new("RGB", imOrigin.size, (255, 255, 255))
    im.paste(imOrigin, mask=imOrigin.split()[3])	

    im = im.convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255)) #creates white canvas of 28x28 pixels
    
    if width > height: #check which dimension is bigger
        #Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0/width*height),0)) #resize height according to ratio width
        if (nheigth == 0): #rare case but minimum is 1 pixel
            nheigth = 1  
        # resize and sharpen
        img = im.resize((20,nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight)/2),0)) #caculate horizontal pozition
        newImage.paste(img, (4, wtop)) #paste resized image on white canvas
    else:
        #Height is bigger. Heigth becomes 20 pixels. 
        nwidth = int(round((20.0/height*width),0)) #resize width according to ratio height
        if (nwidth == 0): #rare case but minimum is 1 pixel
            nwidth = 1
         # resize and sharpen
        img = im.resize((nwidth,20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth)/2),0)) #caculate vertical pozition
        newImage.paste(img, (wleft, 4)) #paste resized image on white canvas
    newImage.save("sample.png")

    tv = list(newImage.getdata()) #get pixel values
    
    tva = [ (255-x)*1.0/255.0 for x in tv] 
    return tva


def main(argv):
    imvalue = imageprepare(argv)
    predint = predictint(imvalue)
    #print predint
    print (predint[0]) #first value in list
    
if __name__ == "__main__":
    main(sys.argv[1])
