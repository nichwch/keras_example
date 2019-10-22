import tensorflow as tf

# loading data from mnist
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
print(x_test.shape)

# normalize input data (pixels now range from 0 to 1)
x_train = x_train / 255.0
x_tet = x_test / 255.0
# # neural network structure
# we're going to flatten our data

layer1 = tf.keras.layers.Flatten()
print(layer1(x_train).shape)

# dense takes takes our flattened image (60000 x 764) and converts it to a (60000 x 200) images
# this is done to get it eventually to 10.
# weights are implicit
# num of rows will always be 60000 because we're predicting 60000 images
layer2 = tf.keras.layers.Dense(200, activation=tf.nn.relu)

# for the next layer, we whittle it down to 150 columns
layer3 = tf.keras.layers.Dense(150, activation=tf.nn.relu)

# softmax function:
# output at every row has 10 items
# creates probability distribution - sum of each row is equal to 1
#
logits_layer = tf.keras.layers.Dense(10,activation=tf.nn.softmax)

model = tf.keras.models.Sequential()
model.add(layer1);
model.add(layer2);
model.add(layer3);
model.add(logits_layer);


# cross entropy measures how close you are to output
# metrics = accuracy tells us how we're judging our model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# train our data set
model.fit(x_train, y_train, epochs=3)
