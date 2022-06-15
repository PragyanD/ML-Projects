import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

def get_dataset(training=True):
    mnist = keras.datasets.mnist #Retrieving MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    if not training:
        return test_images, test_labels
    else:
        return train_images, train_labels

def print_stats(train_images, train_labels):
    num_images = len(train_images)
    image_rows = len(train_images[0])
    image_cols = len(train_images[0][0])
    print(num_images)
    print('{}x{}'.format(image_rows, image_cols))
    #Classifiers for digits
    class_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine']
    _, counts = np.unique(train_labels, return_counts=True)
    for i in range(10):
        print('{}. {} - {}'.format(i, class_names[i], counts[i]))

def build_model():
    #Constructing the model
    model = keras.Sequential()
    model.add(keras.Input(shape=(28,28)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    opt = keras.optimizers.SGD(learning_rate=0.001)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    m=['accuracy']
    model.compile(loss=loss_fn, optimizer=opt, metrics=m)
    return model  
    
def train_model(model, train_images, train_labels, T):
    model.fit(x=train_images, y=train_labels, epochs=T)

def evaluate_model(model, test_images, test_labels, show_loss=True):
    test_loss, test_accuracy = model.evaluate(x =test_images, y=test_labels, verbose=0)
    #Metrics
    if show_loss:
        print('Loss: {:.4f}'.format(test_loss))
    print('Accuracy: {:.2f}%'.format(test_accuracy*100))

def predict_label(model, test_images, index):
    class_names = ['Zero:', 'One:','Two:','Three:','Four:','Five:','Six:','Seven:','Eight:','Nine:']
    f = model.predict(test_images)
    unsortf = f[index]
    usedf = unsortf.copy()
    usedf.sort()
    v1 = usedf[9]
    v2 = usedf[8]
    v3 = usedf[7]
    #Default index values
    i1 = -1
    i2 = -2
    i3 = -3
    for i in range(0,len(unsortf)):
        if(unsortf[i]==v1):
            i1 = i
        if(unsortf[i]==v2):
            i2 = i
        if(unsortf[i]==v3):
            i3 = i
    print(class_names[i1],'{:.2f}%'.format(v1*100))
    print(class_names[i2],'{:.2f}%'.format(v2*100))
    print(class_names[i3],'{:.2f}%'.format(v3*100))




    


