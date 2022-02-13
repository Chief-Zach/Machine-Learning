import pandas as pd
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import torch

tf.compat.v1.ConfigProto()
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

training_df = pd.read_csv(filepath_or_buffer="Banknotes.csv")

float_col = training_df.select_dtypes(include=['float64'])

for col in float_col.columns.values:
    training_df[col] = np.floor(pd.to_numeric(training_df[col], errors='coerce')).astype('float32')

print(training_df.corr())


def build_model(my_learning_rate):
    model = keras.Sequential()

    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, df, feature, label, epochs, batch_size, my_validation_split=0.1):
    """Train the model by feeding it data."""

    # Feed the model the feature and the label.
    # The model will train for the specified number of epochs.
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs, validation_split=my_validation_split)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the error for each epoch.
    hist = pd.DataFrame(history.history)

    # To track the progression of training, we're going to take a snapshot
    # of the model's root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


print("Defined the create_model and training_model functions.")


def plot_the_model(trained_weight, trained_bias, feature, label):
    """Plot the trained model against 200 random training examples."""
    # Label the axes.
    plt.xlabel(feature)
    plt.ylabel(label)

    # Create a scatter plot from 200 random points of the dataset.
    num = 1000
    random_examples = training_df.sample(n=num)
    x = random_examples[feature]
    y = random_examples[label]
    col = []

    # for i in range(0, num):
    #    if y.iloc[i] != 1:
    #        col.append('blue')
    #    else:
    #        col.append('magenta')

    avg_top = []
    avg_bottom = []
    for i in range(0, num):
        if y.iloc[i] != 1:
            a = 'red'
            avg_top.append(y.iloc[i])
        else:
            a = 'blue'
            avg_bottom.append(y.iloc[i])
        plt.scatter(x.iloc[i], y.iloc[i], color=a)

        i += i

    # Create a red line representing the model. The red line starts
    # at coordinates (x0, y0) and ends at coordinates (x1, y1).
    ax = plt.gca()
    x0 = 0
    y0 = trained_bias
    x1 = 10000
    y1 = trained_bias + (trained_weight * x1)
    # plt.plot([x0, x1], [y0, y1], c='r')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-0.2, 1.2])
    avg_top_x = (sum(avg_top)) / len(avg_top)
    avg_bottom_x = (sum(avg_bottom)) / len(avg_bottom)
    top, bottom = [avg_top_x, 1], [avg_bottom_x, 0]
    plt.plot(top, bottom)

    # Render the scatter plot and the red line.
    plt.show()


def plot_the_loss_curve(epochs, rmse):
    """Plot a curve of loss vs. epoch."""

    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min() * 0.97, rmse.max()])
    plt.show()

    print("Defined the plot_the_model and plot_the_loss_curve functions.")


def predicts(num):
    samples = [num]
    samples_to_predict = np.array(samples)
    predictions = my_model.predict(samples_to_predict)
    if predictions[0] >= 0.5:
        print("Real")
    else:
        print("Fake")


learning_rate = 0.01
epochs = 1000
batch_size = 128

# Specify the feature and the label.
my_feature = "Varience"  # the total number of rooms on a specific city block.
my_label = "Class"  # the median value of a house on a specific city block.
# That is, you're going to create a model that predicts house value based
# solely on total_rooms.

# Discard any pre-existing version of the model.
my_model = None
validation_split = 0.2

# Invoke the functions.
my_model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(my_model, training_df,
                                         my_feature, my_label,
                                         epochs, batch_size, validation_split)

print("\nThe learned weight for your model is %.4f" % weight)
print("The learned bias for your model is %.4f\n" % bias)

plot_the_model(weight, bias, my_feature, my_label)
plot_the_loss_curve(epochs, rmse)

data = "weight: " + str(weight) + " bias :" + str(bias) + " epochs: " + str(epochs[-1]) + " learning rate: " + \
       str(learning_rate) + " root-mean-squared " + str(rmse.min())
equation = "%s=%.4fx+%.4f" % (my_label, weight, bias)
print(equation)
file = open("banknote.txt", "a")
file.write(data)
file.write('\n')
file.write(equation)
file.write('\n')
file.close()

samples = [1, 2, 3, 4, 5]

predicts(5)