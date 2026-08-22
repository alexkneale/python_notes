# ============================================================================
# KERAS ARCHITECTURE: CUSTOM LAYERS, SUBCLASSING, AND TRAIN_STEP OVERRIDES
# ============================================================================
# Target Audience: Senior AI Engineers, ML Platform Engineers.
# Topic: Layer Build Deferred State, Functional vs. Subclassing serialization, 
#        overriding Model.train_step, and custom Callbacks.
# Compatibility: Python 3.10+, TensorFlow/Keras 2.10+
# ============================================================================

# ============================================================================
# BEGINNER PRIMER: BASIC CONCEPTS
# ============================================================================
"""
1. What is Keras?
   - Keras is a high-level API designed to make building and training deep learning models 
     fast and easy. It can run on top of different backend engines like TensorFlow, JAX, 
     or PyTorch.
2. What is a Callback?
   - A callback is an object that you pass to `model.fit()`. It hooks into the training 
     loop and automatically triggers custom Python functions at specific times (e.g., at the 
     start of training, after every batch, or at the end of each epoch), allowing you 
     to save checkpointers, log metrics, or halt training early.
3. What is the Functional API vs Model Subclassing?
   - The Functional API lets you construct models like a Lego set, connecting inputs to 
     outputs explicitly (ideal for standard architectures). Model Subclassing gives you 
     complete object-oriented freedom, allowing you to write custom loops, conditional branches, 
     and advanced multi-objective optimization structures.
"""



import tensorflow as tf
from tensorflow import keras
import numpy as np

# ============================================================================
# 1. CUSTOM LAYERS: DEFERRED WEIGHT ALLOCATION IN BUILD()
# ============================================================================
"""
When subclassing `keras.layers.Layer`, you must implement:
1. `__init__`: Saves configuration variables (hyperparameters).
2. `build(input_shape)`: Allocates the layer weights. 
   - WHY defer weight creation to `build()` instead of doing it in `__init__`? 
     Because doing it in `build()` allows shape inference! The layer doesn't need to know 
     the input dimension beforehand; Keras automatically calls `build()` the first time 
     data passes through, dynamically allocating parameter tensors of the correct shapes.
3. `call(inputs)`: Implements the forward propagation logic.
4. `get_config()`: Returns serialization parameters, crucial for saving/loading the model.
"""

class CustomDenseWithWeightNorm(keras.layers.Layer):
    """
    Custom Dense Layer implementing Weight Normalization from scratch:
        W = g * (V / ||V||)
    Where g is a scalar parameter (scale) and V is a parameter matrix.
    """
    def __init__(self, units: int, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        # input_shape is (Batch, features)
        features = input_shape[-1]
        
        # 1. Register parameter matrix V
        self.v = self.add_weight(
            name="weight_v",
            shape=(features, self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        
        # 2. Register parameter scale scalar g
        self.g = self.add_weight(
            name="weight_g",
            shape=(self.units,),
            initializer="ones",
            trainable=True
        )
        
        # 3. Register bias b
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        
        # Complete build lifecycle
        super().build(input_shape)

    def call(self, inputs):
        # Calculate normalized weight: W = g * V / ||V||
        v_norm = tf.linalg.l2_normalize(self.v, axis=0)
        weights = self.g * v_norm
        
        # Equation: y = xW + b
        outputs = tf.matmul(inputs, weights) + self.bias
        
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        # Essential to enable serialization and deserialization via model load APIs!
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": keras.activations.serialize(self.activation)
        })
        return config


# ============================================================================
# 2. OVERRIDING TRAIN_STEP IN CUSTOM MODELS
# ============================================================================
"""
For complex, multi-objective structures (like Generative Adversarial Networks, 
Variational Autoencoders, or custom gradient penalties), Keras allows you to 
subclass `keras.Model` and override the `train_step(self, data)` method.
- This lets you run custom training math under the hood of `model.fit()`, 
  maintaining native support for compiled callbacks, evaluation, and logging.
"""

class CustomTrainerModel(keras.Model):
    def __init__(self, encoder_layer, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder_layer
        # We manually register loss metrics to compile reports
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def train_step(self, data):
        # Data passed from fit() is usually a tuple (inputs, targets)
        x, y_true = data

        # Record gradient computations
        with tf.GradientTape() as tape:
            y_pred = self.encoder(x, training=True)
            # Custom mean squared error calculation
            loss = tf.reduce_mean(tf.square(y_pred - y_true))

        # Retrieve gradients and update weights
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update running metrics
        self.loss_tracker.update_state(loss)

        # Return a dict mapping metric names to current values
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        # Listing metrics here allows fit() to automatically reset them at epoch boundaries!
        return [self.loss_tracker]


# ============================================================================
# INTERVIEW PREPARATION: QUESTIONS & LIVE CODING EXERCISES
# ============================================================================
"""
A. CONCEPTUAL QUESTIONS

1. EASY: Compare Keras's Sequential API vs. Functional API.
   - Answer: Sequential is a linear stack of layers, extremely simple to write but 
     incapable of handling non-linear structures. The Functional API models models 
     as directed acyclic graphs (DAGs), natively supporting multiple inputs, multi-outputs, 
     skip-connections, and layer sharing by defining symbolic tensor inputs and outputs.

2. MEDIUM: Why must you implement `get_config()` when writing a custom layer in Keras?
   - Answer: Keras uses `get_config()` to support model serialization (saving models 
     as H5 or SavedModel formats). `get_config()` returns a Python dictionary containing 
     all the initialization parameters of the layer. When loading the model, Keras passes 
     this dictionary as `**kwargs` to the layer's `from_config()` method to reconstruct it.

3. HARD: What is the benefit of overriding `train_step()` in a custom Keras Model 
   instead of writing a raw TensorFlow training loop from scratch?
   - Answer: Writing a raw training loop bypasses Keras, forcing you to manually code 
     epoch iterations, training/evaluation toggling, distributed strategy configurations, 
     metric tracking, and callback lifecycles. Overriding `train_step()` delegates 
     all these structural utilities (distributed CPU/GPU scaling, early stopping callbacks, 
     tensorboard, validation loops) to Keras's robust `fit()` engine while giving you 
     complete control over the gradient mathematics.

B. LIVE CODING EXERCISES

1. EXERCISE: Implement a custom Keras callback `LearningRateSchedulerCallback` that monitors 
   the training loss at the end of each epoch, and if the loss is below a specified threshold, 
   manually decays the optimizer's learning rate by a factor of 10.
   - Code Template to execute:
"""

class LearningRateSchedulerCallback(keras.callbacks.Callback):
    def __init__(self, threshold: float, decay_factor: float = 0.1):
        super().__init__()
        self.threshold = threshold
        self.decay_factor = decay_factor

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get("loss")
        
        if current_loss is not None and current_loss < self.threshold:
            # Retrieve optimizer from the model
            optimizer = self.model.optimizer
            # Read current learning rate
            old_lr = float(keras.backend.get_value(optimizer.lr))
            # Calculate and set updated learning rate
            new_lr = old_lr * self.decay_factor
            keras.backend.set_value(optimizer.lr, new_lr)
            print(f"\n[Callback] Epoch {epoch}: Loss ({current_loss:.4f}) is below threshold ({self.threshold:.4f}). Decaying learning rate from {old_lr:.6f} to {new_lr:.6f}.")


# ============================================================================
# EXECUTING DEMONSTRATIONS
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEMONSTRATING CUSTOM WEIGHT-NORMALIZED LAYER & TRAINING")
    print("="*80)

    # 1. Instantiate custom layer
    custom_layer = CustomDenseWithWeightNorm(units=1, activation="sigmoid")
    
    # 2. Build custom training model using the layer
    model = CustomTrainerModel(encoder_layer=custom_layer)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1))

    # 3. Create synthetic regression data
    x_train = np.random.randn(100, 8).astype(np.float32)
    y_train = (x_train[:, 0] + x_train[:, 1] > 1.0).astype(np.float32).reshape(-1, 1)

    # 4. Fit using our custom trainer model and our custom Callback
    lr_callback = LearningRateSchedulerCallback(threshold=0.1, decay_factor=0.1)
    
    # Run a quick 3 epoch training pass to verify compilation and execution
    history = model.fit(x_train, y_train, epochs=3, batch_size=32, callbacks=[lr_callback], verbose=1)