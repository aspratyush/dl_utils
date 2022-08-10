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

- Saving/Loading using SavedModel format (preferred)
  - `tf.saved_model.save(model, "path")`
  - contains both the graph (`.pb` file) and weights (in `variables/`).
  - `tf.saved_model.load("path")` loads the graph without code.