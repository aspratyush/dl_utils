import tensorflow as tf
import tensorflow.experimental.numpy as tnp
import numpy as np
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

tnp.experimental_enable_numpy_behavior()

x = tf.linspace(-2, 2, 201)
x = tf.cast(x, tf.float32)

m = 3
c = 5
def f(x):
    return m*x + c

y = f(x)
noise = tf.random.normal(shape=tf.shape(x))
y_noise = f(x) + noise

class MyModel(tf.Module):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be randomly initialized
    self.w = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def __call__(self, x):
    return self.w * x + self.b

class MyModelKeras(tf.keras.Model):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be randomly initialized
    self.w = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def call(self, x):
    return self.w * x + self.b

use_keras_model = True
use_keras_training = True

if use_keras_model:
  model = MyModelKeras()
else:
  model = MyModel()


if use_keras_training:
  ########### Keras Style ##########
  # compile sets the training parameters
  model.compile(
      # By default, fit() uses tf.function().  You can
      # turn that off for debugging, but it is on now.
      run_eagerly=False,

      # Using a built-in optimizer, configuring as an object
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),

      # Keras comes with built-in MSE error
      # However, you could use the loss function
      # defined above
      loss=tf.keras.losses.mean_squared_error,
  )
  model.fit(x, y, epochs=10, batch_size=1000)
  ##################################
else:
  def loss(target_y, predicted_y):
    return tf.reduce_mean(tf.square(target_y - predicted_y))

  # Given a callable model, inputs, outputs, and a learning rate...
  def train(model, x, y, learning_rate):

    with tf.GradientTape() as t:
      # Trainable variables are automatically tracked by GradientTape
      current_loss = loss(y, model(x))

    # Use GradientTape to calculate the gradients with respect to W and b
    dw, db = t.gradient(current_loss, [model.w, model.b])

    # Subtract the gradient scaled by the learning rate
    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)

  # Collect the history of W-values and b-values to plot later
  weights = []
  biases = []
  epochs = range(10)

  # Define a training loop
  def report(model, loss):
    return f"W = {model.w.numpy():1.2f}, b = {model.b.numpy():1.2f}, loss={loss:2.5f}"


  def training_loop(model, x, y):

    for epoch in epochs:
      # Update the model with the single giant batch
      train(model, x, y, learning_rate=0.1)

      # Track this before I update
      weights.append(model.w.numpy())
      biases.append(model.b.numpy())
      current_loss = loss(y, model(x))

      print(f"Epoch {epoch:2d}:")
      print("    ", report(model, current_loss))

  current_loss = loss(y, model(x))

  print(f"Starting:")
  print("    ", report(model, current_loss))

  training_loop(model, x, y)


  # plt.plot(x, y_noise, '.', label="Data")
  # plt.plot(x, f(x), label="Ground truth")
  # plt.plot(x, model(x), label="Predictions")
  # plt.legend()
  # plt.show()
  # print("Current loss: %1.6f" % loss(y, model(x)).numpy())

  plt.plot(epochs, weights, label='Weights', color=colors[0])
  plt.plot(epochs, [m] * len(epochs), '--',
          label = "True weight", color=colors[0])

  plt.plot(epochs, biases, label='bias', color=colors[1])
  plt.plot(epochs, [c] * len(epochs), "--",
          label="True bias", color=colors[1])
  plt.legend()
  plt.show()

  plt.plot(x, y_noise, '.', label="Data")
  plt.plot(x, f(x), label="Ground truth")
  plt.plot(x, model(x), label="Predictions")
  plt.legend()
  plt.show()
  print("Current loss: %1.6f" % loss(y, model(x)).numpy())