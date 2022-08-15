## Debugging

#### Device placement
- Turn on device placement logging : `tf.debugging.set_log_device_placement(True/False)`
- link : https://www.tensorflow.org/api_docs/python/tf/debugging/set_log_device_placement


## Automatic differentiation

#### Gradients for `tf.Variables`
- `tf.GradientTape.gradient(target, sources)` computes the `d_target/d_sources`
- accepts single or nested source.
  - supported nest types include a list and a dictionary
  - intermediate outputs can also be sources
- "watch" the sources (if `tf.Tensor`) by calling context managers `watch`
```
    x = tf.Tensor([3])
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = x**x
    dy_dx = tape.gradient(y, x)
```
- list the watched variables using `tape.watched_variables()`
- Unconnected gradients
  - returns `None` for unconnected gradient scenarios : https://www.tensorflow.org/guide/autodiff#cases_where_gradient_returns_none (i.e., Tensor, outside graph op, integer/string, stateful objects)
  - override this behaviour by passing `unconnected_gradients=tf.UnconnectedGradients.ZERO)` to `tape.gradient()` call.
- link : https://www.tensorflow.org/guide/autodiff#gradient_tapes

#### Gradients for a model
- `tf.Variables` are auto collected into a `tf.Module` or its subclasses (`keras.Model`, `layers.Layer`) in `Module.trainable_variables` property.
  - disable this behaviour by passing arg `watch_accessed_variables=False` when creating the tape context manager. 
- link: https://www.tensorflow.org/guide/autodiff#gradients_with_respect_to_a_model

#### Gradient for nested targets / Jacobians
- nested targets results in sum of gradients of targets for each source.
- For Jacobian calculation, invoke `tape.jacobian`. Shape will be of the target.

#### Unregistered gradient for `raw_ops`
- `raw_ops` do not have a gradient implemented, and will throw and error.
- If backprop is needed, implement the gradient and register using `tf.RegisterGradient` 
- link: https://www.tensorflow.org/guide/autodiff#no_gradient_registered


## TF Graph

- `tf.Graph` is a data structure that contains a set of `tf.Operations` (units of compute), and `tf.Tensor` objects (units of data that flows between ops)
- Since this is a data structure, it allows for portability across platforms.
- Graph optimization library (Grappler) : https://www.tensorflow.org/guide/graph_optimization

#### `tf.function`
- `tf.function` takes a regular Py function, and returns a TF `Function`
- top-level function will cause graph conversion for all inner functions.
- Process followed:
  - Trace Py code
  - convert to graph code
  - link: https://www.tensorflow.org/guide/function
- `tf.autograph` used to convert Py code into graph code.
  - link: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/index.md
  - `tf.autograph.to_code(py_function)` to view the converted Py code
  - Show actual graph : `py_function.get_concrete_function(tf.constant(1)).graph.as_graph_def()`
  - 1 graph per `input_signature`, and hence is polymorphic - https://www.tensorflow.org/guide/intro_to_graphs#polymorphism_one_function_many_graphs
  - show all the graphs : `py_function.pretty_printed_concrete_signatures()`
- Use `tf.config.run_functions_eagerly(True)` to verify if Function is working correctly.
    - this turns off Function's ability to create and run graphs
- `tf.print` can be used in both Eager and Function mode. `print` will be called only once when tracing happens.
- New Python arguments always trigger the creation of a new graph.
- **NOTE**: Only needed operations are run during graph execution, and an error is not raised. Do not rely on an error being raised while executing a graph.
- better performance : https://www.tensorflow.org/guide/function


#### `tf.Module`
- `Module`   
     |-------> `layers.Layer` : use `call()`   
     |-------> `keras.Model` : use `call()`  
     |-------> `submodules` : use `__call__`
  - submodules (subclasses) can be queried using : `model.submodules`
  - variables : `model.variables`, trainable_variables : `model.trainable_variables`
  - Overriding `tf.keras.Model` is a very Pythonic approach to building TensorFlow models.
  - link : https://www.tensorflow.org/guide/intro_to_modules#defining_models_and_layers_in_tensorflow

- Saving/Restoring the model using checkpoint
  - `checkpt = tf.train.Checkpoint(model); checkpt.write(path)` creates a checkpoint of the model.
  - `checkpt = tf.train.Checkpoint(model); checkpt.restore(path)` restores from a checkpoint.
  - helper class : `tf.checkpoint.CheckpointManager`
  - - link : https://www.tensorflow.org/guide/checkpoint

- Saving/Loading using SavedModel format (preferred#1)
  - `tf.saved_model.save(model, "path")`
  - contains both the graph (`.pb` file) and weights (in `variables/`).
  - `tf.saved_model.load("path")` loads the graph without code.

- Saving/Loading using keras APIs (preferred#2)
  - - `tf.keras.models.save()` : saves model arch, weights, optimizer state and args passed to compile.
  - enables restarting training from the last point
  - `tf.keras.models.load_model()` : loadd all saved contents from the model.


### Ops

#### Reshape
- `tf.reshape` is fast since a new Tensor is created pointing to same memory.
- TF uses row-major indexing
    - link : https://www.tensorflow.org/guide/tensor#manipulating_shapes
#### String
- `tf.string` is byte type, not a unicode string.
- working with Unicode text : https://www.tensorflow.org/text/guide/unicode
- call `tf.io.decode_` apis to convert byte strings to numbers.


## Keras Model
- `Sequential` model : 
  - 1 I/P Tensor --> stack of layers --> 1 O/P Tensor
  - not appropriate for non-linear topology (residual / multi-branch / multi i/o).
  - weights are available once I/P shape is known
- custom training : https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough

- `Functional` model:
  - can handle models with non-linear topology, shared layers, and even multiple inputs or outputs.
  - can re-use sub-graphs in multiple models.
- models can be nested. models are callable, similar to layers(link : https://www.tensorflow.org/guide/keras/functional#all_models_are_callable_just_like_layers)
- supports nested loss (list / dict)


### extract features
```
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)
```
OR
```
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=initial_model.get_layer(name="my_intermediate_layer").output,
)
```

### Custom loss
- link: https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_losses
#### Method#1
- define a function that accepts a batch of data, and follows this pattern:   
```
def custom_loss(y_true, y_pred):
  return ...
```

#### Method#2 - with params
- if extra params are needed, best way is to subclass `tf.keras.losses.Loss`
- pass params in `__init__`, and implement `call()`
```
class CustomMSE(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor
```

#### Method#3 - `self.add_loss()` 
- add in `call()` or add to model when using functional API. 


### Custom Metrics
- link: https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_metrics
- subclass `tf.keras.metrics.Metric`, and implement `__init__`, `reset_state`, `result` and `update_state`
- another option is to use `self.add_metric()`, or add it to the model in functional API.
