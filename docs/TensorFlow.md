## Debugging

#### Better debugging with `tf.function`
- eager execution is turned on by default.
- Use `tf.config.run_functions_eagerly(True)` to verify if `Function` is working correctly.
  - this turns off `Function`'s ability to create and run graphs
- Always debug in Eager mode. Then decorate with `@tf.function`.
  - Numpy and Py ops are converted to const ops.
- Don't rely on pure Python ops (object mutations / list appends)
- Use `tf.debugging.enable_check_numerics` and `tf.debugging.*`.
- link: https://www.tensorflow.org/guide/function#debugging

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
- It is the raw, language-agnostic, portable representation of a TensorFlow computation.
- Graph optimization library (Grappler) : https://www.tensorflow.org/guide/graph_optimization
- `Function` manages a cache of `ConcreteFunctions` and picks the right one for your inputs. `tf.function` wraps a Python function, returning a Function object. Tracing creates a `tf.Graph` and wraps it in a `ConcreteFunction`.

#### AutoGraph
- on by default. Converts a subset of eager Py code to graph compatible TF Ops (including `if, for, while`).
- Control flow is easier to read if written in Py.
- Conditionals:
  - `if` statements are converted to `tf.cond` if condition is a Tensor.
  - `for` and `while` are converted to `tf.while_loop`.
    - if range is a Tensor / `tf.data.Dataset`.
    - Python loop executes during tracing, and adds ops per iteration to the Graph.
- Looping over Python data:
  - wrap python data in `tf.data.Dataset`. When using `from_generator`, data is fetched from Py via `tf.py_function`,having PERFORMANCE implications, when using `from_tensor`, copy of data is bundled as a `tf.constant`, having MEMORY implications.
  - Reading from `TFRecordDataset` `CsvDataset`, etc. is the most effective way to consume data, as TF manages async loading and prefetching, without involving Python.
  - link: https://www.tensorflow.org/guide/data
- Accumulation in a loop:
  - List is a Python object. Use `tf.TensorArray`.
- Limitations:
  - link: https://www.tensorflow.org/guide/function#limitations
  - executing Python side-effects. Could try wrapping them in `tf.py_function`, but it doesnt work well in distributed training.
  - Avoid mutating Python objects that live outside the `Function`.
  - Avoid using Python iterators. use `tf.data` from iterator patterns.
  - `Function` must return all its outputs. Otherwise, a leak may happen.
  - recursive `Function` is not supported.

#### `tf.function`
- `tf.function` takes a regular Py function, and returns a TF `Function`
- top-level function will cause graph conversion for all inner functions.
- Creates a Python independant data flow graph. Enables creating performant and portable models.
- Process followed:
  - Trace Py code
  - convert to graph code
  - link: https://www.tensorflow.org/guide/function
- `tf.autograph` used to convert Py code into graph code.
  - link: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/index.md
  - `tf.autograph.to_code(Function.python_function)` to view the converted Py code
  - Show actual graph : `Function.get_concrete_function(tf.constant(1)).graph.as_graph_def()`
  - 1 graph per `input_signature`, and hence is polymorphic - https://www.tensorflow.org/guide/intro_to_graphs#polymorphism_one_function_many_graphs
  - show all the generated graph types : `Function.pretty_printed_concrete_signatures()`
- `tf.print` can be used in both Eager and Function mode. `print` will be called only once when tracing happens.
- New Python arguments always trigger the creation of a new graph.
- **NOTE**: Only needed operations are run during graph execution, and an error is not raised. Do not rely on an error being raised while executing a graph.
- better performance : https://www.tensorflow.org/guide/function
- Use `timeit` to see performance improvement:
```
import timeit
conv_layer = tf.keras.layers.Conv2D(100, 3)
@tf.function
def conv_fn(image):
  return conv_layer(image)
image = tf.zeros([1, 200, 200, 100])
print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
```

#### Tracing
- Optional Stage1 : `tf.function` creates a new `tf.Graph`. All TF ops are deferred, and captured by the graph. Py code runs normally.
- Stage2 : everthing that was deferred is run.
- Speed improvement seen by skipping 1st stage, and executing 2nd stage.
- Upon repeatedly calling a `Function` with the same argument type, TensorFlow will skip the tracing stage and reuse a previously traced graph. Both stages invoked with new arg dtype.
- Rules: https://www.tensorflow.org/guide/function#rules_of_tracing
- **NOTE**: If `Function` retraces a new graph for every call, code would execute more slowly than if `tf.function` wasn't used!
- To control tracing behaviour, pass a fixed `input_signature` with `None` shape, pass `Tensor` objects instead of Python args:
  - `None` shape addresses variable length I/P (e.g. Transformer and Deep Dream)
```
@tf.function(input_signature=(tf.TensorSpec(shape=[None], dtype=tf.int32),))
def next_collatz(x):
  print("Tracing with", x)
  return tf.where(x % 2 == 0, x // 2, 3 * x + 1)
```
- Get concrete function with the below. Python type in concrete type call is tread as a constant. 
```
@tf.function
def pow(a, b):
  return a ** b

square = pow.get_concrete_function(a=tf.TensorSpec(None, tf.float32), b=2)
print(square)
```


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
  - multi input multi output example : https://www.tensorflow.org/guide/keras/train_and_evaluate#passing_data_to_multi-input_multi-output_models

-  Custom model/layer handling
   -  Add the following member functions.
```
def get_config(self):
    return {"hidden_units": self.hidden_units}

@classmethod
def from_config(cls, config):
    return cls(**config)
```

- Custom functions handling
  - Need it defined.

### Extract features
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
- useful when loss is based on internal states of a subclassed layer / model (e.g. : KL divergence).
- link: https://www.tensorflow.org/guide/keras/custom_layers_and_models#putting_it_all_together_an_end-to-end_example


### Custom Metrics
- link: https://www.tensorflow.org/guide/keras/train_and_evaluate#custom_metrics
- subclass `tf.keras.metrics.Metric`, and implement `__init__`, `reset_state`, `result` and `update_state`
- another option is to use `self.add_metric()`, or add it to the model in functional API.


### Data Pipelines

#### Method 1 : `np.array`
- good for small datasets

#### Method 2 : `tf.data.Dataset`
- good for wide range of data especially when a lot of Py preprocessing is not needed.

#### Method 3 : subclass `keras.utils.Sequence`
- link : https://www.tensorflow.org/guide/keras/train_and_evaluate#using_a_kerasutilssequence_object_as_input
- good for large data which need many Py side pre-processing.
- works well with multiprocessing, can be shuffled.
- subclass needs to implement `__getitem__` and `__len__`.
  - `__len__` : return `number of items/batch_size` (i.e., `num_steps_per_epoch`)
  - `__getitem__` : return idx based data
  - `on_epoch_end` : OPTIONAL. Could be used to do post epoch handling.
```
class CIFAR10Sequence(tf.keras.utils.Sequence):
    def __init__(self, filenames, labels, batch_size):
        self.filenames, self.labels = filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([
            resize(imread(filename), (200, 200))
               for filename in batch_x]), np.array(batch_y)
```

### Sample / Class Weights
- these modulate the contribution of each sample / class in overall loss.
- by default, sample weight is dependant on data frequency in the dataset.
- `class_weight` accepts a dict with class weights, pass to `model.fit()`.
  - used to balance classes without resampling / give more importance to one class.
- `sample_weight` is more fine-grained, giving per sample weights.
  - When using `tf.data` or generator, yield `(X,y,sample_weights)`
  - can even mask entire class if weights used are 0 and 1.

### Loss weight
- used to give weighted importance to loss terms in multi-output graphs.
- sometimes, some o/ps can have `None` loss. These heads get used for prediction but not for training.


### Callbacks
- link: https://www.tensorflow.org/guide/keras/custom_callback
- `ModelCheckpoint` can be used to store / restore models.
```
import os

# Prepare a directory to store all the checkpoints.
checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


def make_or_restore_model():
    # Either restore the latest model, or create a fresh one
    # if there is no checkpoint available.
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    return get_compiled_model()


model = make_or_restore_model()
```

## Customize keras based models

### Customize Layer / Model class
- encapsulates state (weights) defined in `__init__` and transformation from inputs to outputs captured in `call()`.
  - variables in `__init__` get tracked as layer's `weights`
  ```
  # check if weights are getting tracked
  assert linear_layer.weights == [linear_layer.w, linear_layer.b]
  # get trainable weights
  print(linear_layer.trainable_weights)
  # get non trainable weights
  print(linear_layer.nontrainable_weights)
  ```
  - either define the weights as `tf.Variable` or by using `self.add_weight()`.
  - non trainable weights can be defined by passing `trainable=False` to `tf.Variable` construction.
  - use `assign_` in `call()` if need to update weights.
  - Privileged training arg : if a layer has different behaviour in training and at inference, add an extra arg in `call()` to capture mode.
  - Privileged mask arg : used in RNNs to mask certain time steps. Extra arg in `call()`.
  - if defined recursively (i.e., layer inside layer), outermost layer will track weights for all wrapped layers.
- **OPTIONAL** : Best practice is to add `build(self, input_shape)` which captures lazy weight creation upon knowing shape of inputs.
- **NOTE** : any variable creation in `__call__` should be wrapped in `tf.init_scope`.


### Customize `model.fit()`
- link: https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#a_first_simple_example
- same concepts as below apply to `evaluate()`, by overriding `test_step`.
- full example : https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit#wrapping_up_an_end-to-end_gan_example
#### Step 1
- subclass `keras.Model`
- implement `train_step(self, data)`. Inside this, call :
  - `self.compiled_loss(y, y_pred, regularization_losses=self.losses)`
  - calculate gradients on trainable params.   
  - `self.optimizer.apply_gradients(zip(gradients, trainable_vars))`
  - `self.compiled_metrics.update_state(y, y_pred)`
  - Return a dict mapping metric names to current value, i.e., `return {m.name: m.result() for m in self.metrics}`

#### Step 2 (if loss is not defined in compile)
- have a `Metric` instance to track loss and score.
- `train_step()` computes per step loss, updates metric states by calling `update_state()`, queries them via `result()` to be returned in a dict.
- either call `reset_states()` manually before every epoch, or implement a metrics property which returns the metrics in use. This way a call to `fit()` will auto-call `reset_states()`.

#### Step 3 : support `sample_weight` and `class_weight`
- unpack sample_weight from data arg passed to `train_step`.
- pass it to `compiled_loss` and `compiled_metrics`.


### Implement custom training
- define model. Instantiate the metric at the start of the loop.
- For loop over epochs. For each epoch, iterate over the dataset, in batches.
- For each batch, we open a `GradientTape()` scope, and call the model (forward pass) and compute the loss.
- Outside the scope, retrieve the gradients of the weights of the model with regard to the loss.
- Finally, use the optimizer to update the weights of the model based on the gradients.
- Call `metric.update_state()` after each batch. Call `metric.result()` to display the current value of the metric.
- Call `metric.reset_states()` to clear the state of the metric (typically at the end of an epoch).
- link : https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch#end-to-end_example_a_gan_training_loop_from_scratch
- core example : https://www.tensorflow.org/guide/core/quickstart_core
