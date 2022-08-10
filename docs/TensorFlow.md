## Debugging

### Device placement
- Turn on device placement logging : `tf.debugging.set_log_device_placement(True/False)`
- link : https://www.tensorflow.org/api_docs/python/tf/debugging/set_log_device_placement


### Automatic differentiation
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