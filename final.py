import cv2
import os,copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from datetime import datetime

#%%
Train_PATH='./train/anpr_ocr/train/img'
Test_PATH='./test/anpr_ocr/test/img'
def get_image_information(PATH):
    f0=os.listdir(PATH)
    im = cv2.imread(PATH+'/'+f0[0],0)
    Height,Width = im.shape
    im_num = len(os.listdir(PATH))
    #images_pool
    images_pool = np.zeros((im_num,Height,Width))
    #labels_pool
    labels_pool=[]
#    print(images_pool.shape)
    ii = 0
    for f in os.listdir(PATH):
        PATH_img = PATH+'/'+f
        im = cv2.imread(PATH_img,0)
        images_pool[ii] = im
        ii+=1
        labels_pool.append(f[:8])
    
    return images_pool, labels_pool

def clean(img):
    im = copy.deepcopy(img)
    im=(255-im)
    im[0:4,:]=0
    im[30:,:]=0
    im[:,:4]=0
    im[:,-4:]=0
    im[im<200]=0
    return im

def get_loc_x(img,show=False):
    nrow,ncol = img.shape
    im = clean(img)
    W_cal = np.zeros(ncol)
    
    for j in range(ncol):
        for i in range(nrow):
            if(im[i][j]>0):
                W_cal[j] += 1  
    W_cal[W_cal<2]=0
    W_cal[W_cal>25]=0
    loc_x = []  
    for i in range(len(W_cal)-1):
        if(W_cal[i]==0 and W_cal[i+1]>0):
            loc_x.append(i)
        if(W_cal[i]>0 and W_cal[i+1]==0):
            loc_x.append(i+2)
    loc_x = np.reshape(loc_x,(-1,2))  
    if(show):
        print('We calculate the vertical Histagram of the license image')
        print('then by the get_loc_x() function We can get each font\'s x-axis location' )
        print('so We can dig out the font patches')
    for i in range(loc_x.shape[0]):
        patch = im[:,loc_x[i][0]:loc_x[i][1]]
        if(show):
            plt.imshow(patch,cmap='gray')
            plt.show()
        
    return loc_x

def get_loc_y(img,loc_x):
    im = clean(img)
    nrow,ncol = im.shape
    loc_y = []
    for i in range(len(loc_x)):
        patch = im[:,loc_x[i][0]:loc_x[i][1]]
        patch = np.array(patch)
        nrow,ncol = patch.shape
        H_cal = np.zeros(nrow)      
#        plt.imshow(patch,cmap='gray')
#        plt.show() 
        for j in range(nrow):
            for k in range(patch.shape[1]):
                if(patch[j,k]>1):
                    H_cal[j]+=1
        count = 0 
        for ii in range(len(H_cal)-1):
            if(H_cal[ii]==0 and H_cal[ii+1] >0):
                loc_y.append(ii)
                count+=1
            if(H_cal[ii]>0 and H_cal[ii+1]==0):
                loc_y.append(ii+2)
                count+=1
            if(count>1):
                break  
            
    loc_y = np.reshape(loc_y,(-1,2)) 
    return loc_y
def get_images(img,show=False):
    loc_x= get_loc_x(img,show)
    loc_y = get_loc_y(img,loc_x)
    if(show):
        print('However, We still need to get the y-axis location of each font')
        print('Then,I calculate the horizontal Histagram of the license image')
        print('the location of each font have already done, and resize each font patch into size(20,20)')
    loc = np.zeros((len(loc_y),4))
#    print(loc.shape)
#    print(loc_x.shape)    
#    print(loc_y.shape)
    patches =[]
    for i in range(len(loc)):
        loc[i][0] = loc_y[i][0]
        loc[i][1] = loc_y[i][1]
        loc[i][2] = loc_x[i][0]
        loc[i][3] = loc_x[i][1]
        y1 = loc_y[i][0]
        y2 = loc_y[i][1]
        x1 = loc_x[i][0]
        x2 = loc_x[i][1]
        
        patch = copy.deepcopy(img[y1:y2,x1:x2])
        patch = cv2.resize(patch,(20,20))
        if(show):
            plt.imshow(patch,cmap='gray')
            plt.show()
        patches.append(patch)

    return patches

#%%
 
def Encoder(a):
    if(a<='9'):
        return int(a)
    if(a>='A'):
        return 10+ord(a)-ord('A')

def Decoder(a):
    if a < 10:
        return str(a)
    else:
        return chr(a-10+ord('A'))
    
def get_data(PATH):
    train_images_pool, train_labels_pool = get_image_information(PATH)
    img_pool = []
    for i in range(len(train_images_pool)):
#        print(i)
        patches = get_images(train_images_pool[i])
        for j in patches:
            img_pool.append(j)
        
    label_pool = []
    for label in train_labels_pool:
        for ii in range(len(label)):
            label_pool.append(Encoder(label[ii]))
    
    return img_pool,label_pool
    
img_pool,label_pool = get_data(Train_PATH)
img1_pool,label1_pool = get_data(Test_PATH)
    
#%%
    
X_train = np.array(img_pool)
Y_train = label_pool 
X_test = np.array(img1_pool)
Y_test = label1_pool       

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

learning_phase = keras.backend.learning_phase()

keras_model = Sequential()
keras_model.add(Conv2D(32, (5, 5), input_shape=(20, 20, 1),padding = 'SAME'))
keras_model.add(Activation('relu'))
keras_model.add(Conv2D(32, (5, 5)))
keras_model.add(Activation('relu'))
keras_model.add(MaxPooling2D(pool_size=(2, 2)))
keras_model.add(Conv2D(64, (3, 3)))
keras_model.add(Activation('relu'))
keras_model.add(Conv2D(64, (3, 3)))
keras_model.add(Activation('relu'))
keras_model.add(MaxPooling2D(pool_size=(2, 2)))
keras_model.add(Flatten())
keras_model.add(Dense(200))
keras_model.add(Activation('relu'))
keras_model.add(Dropout(0.4))
keras_model.add(Dense(200))
keras_model.add(Activation('relu'))
keras_model.add(Dense(36))


x_input = tf.placeholder(tf.float32, shape=[None, 20,20,1])
y_input = tf.placeholder(tf.int64, shape=None)
# use the build keras model as a single operation
pre_softmax = keras_model(x_input)
# add loss to the network
y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pre_softmax, labels=y_input)
xent = tf.reduce_sum(y_xent, name='y_xent')
# accuracy
predictions = tf.argmax(pre_softmax, 1)
correct_prediction = tf.equal(predictions, y_input)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# add optimizer
global_step = tf.train.get_or_create_global_step()
train_step = tf.train.AdamOptimizer(0.01).minimize(xent, global_step=global_step)

# test_data feed_dict
test_dict = {x_input: X_test, y_input: Y_test, learning_phase: 0}

TRAIN_STEPS = 500
BATCH_SIZE = 200
OUTPUT_STEPS = 10

#we save the model to H5 ， Uncomment and Run to see the whole process
#
#with tf.Session() as sess:
#    # Initialization
#    sess.run(tf.global_variables_initializer())
#
#    # Main training loop
#    for ii in range(TRAIN_STEPS):
#        # get train batch data
#        batch_start = (ii * BATCH_SIZE) % len(Y_train)
#        batch_end = batch_start + BATCH_SIZE
#        if batch_end >= len(Y_train):
#            continue
#        x_batch = X_train[batch_start:batch_end]
#        y_batch = Y_train[batch_start:batch_end]
#
#        # prepare train feed_dict
#        train_dict = {x_input: x_batch, y_input: y_batch, learning_phase: 1}
#
#        # train the network
#        sess.run(train_step, feed_dict=train_dict)
#
#        # Output to stdout
#        if ii % OUTPUT_STEPS == 0:
#          test_acc = sess.run(accuracy, feed_dict=test_dict)
#          train_acc = sess.run(accuracy, feed_dict=train_dict)
#          print('Step {}:    ({})'.format(ii, datetime.now()))
#          print(' training nat accuracy {:.4}%'.format(train_acc * 100))
#          print(' testing accuracy {:.4}%'.format(test_acc * 100))
#
#
#    
#    keras_model.save_weights('my_keras_model_weights.h5')


#%%
'''
show
'''
VAL_PATH = './val/anpr_ocr/train/img/'
pool,lab = get_image_information(VAL_PATH)
imgs = pool[0]
labs = lab[0]

print('\n\nHere is the license image:\n')
plt.imshow(pool[0],cmap='gray')
plt.show()
print('the license is',labs)
patches = get_images(imgs,show = True)


#%%

VAL_PATH = './val/anpr_ocr/train/img/'
img2_pool,label2_pool = get_data(VAL_PATH)
X_val = np.array(img2_pool)
Y_val = label2_pool
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))
X_val = X_val.astype('float32')
X_val /= 255
Val_dict = {x_input: X_val, y_input: Y_val, learning_phase: 0}
ans=[]
accu=0
with tf.Session() as sess:
    keras_model.load_weights('./my_keras_model_weights.h5')
    test_acc = sess.run(accuracy, Val_dict)
    accu=test_acc
    test_pre = sess.run(predictions, Val_dict)
    print('Step {}:    ({})'.format(0, datetime.now()))
    print(' testing accuracy {:.4}%'.format(test_acc * 100))
    
    for i in test_pre:
        ans.append(Decoder(i))
imgs = pool[1]
plt.imshow(imgs,cmap='gray')
plt.show()   
print(' testing accuracy {:.4}%'.format(accu * 100))
print(ans[8:16])