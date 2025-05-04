# -*- coding: utf-8 -*-
"""
Created on Sat May  3 22:21:31 2025

@author: Owner
"""

# Add these functions after the train_test function to calculate and print validation metrics

def print_validation_metrics(model, val_data, val_labels, model_name):
    """
    Print validation metrics for a given model
    """
    if model_name == "MLP" or model_name == "CNN":
        # For classification models
        val_accuracy = model.history.history['val_accuracy'][-1]
        val_loss = model.history.history['val_loss'][-1]
        print(f"\n{model_name} Validation Metrics:")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # If we have validation data explicitly provided
        if val_data is not None and val_labels is not None:
            acc, prec, rec, f1 = model.perf(val_data, val_labels)
            print(f"Validation Performance Metrics:")
            print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
            
    elif model_name == "DAE":
        # For autoencoder model
        val_loss = model.history.history['val_loss'][-1]
        print(f"\n{model_name} Validation Metrics:")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # If we have validation data explicitly provided
        if val_data is not None and val_labels is not None:
            acc, prec, rec, f1 = model.perf(val_data, val_labels)
            print(f"Validation Performance Metrics:")
            print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")

# Function to create validation splits
def create_validation_split(train_data, train_labels, val_split=0.1, random_state=42):
    """
    Create a validation split from training data
    """
    train_idx, val_idx = train_test_split(
        np.arange(len(train_data)), 
        test_size=val_split, 
        random_state=random_state, 
        stratify=np.argmax(train_labels, axis=1) if len(train_labels.shape) > 1 else train_labels
    )
    
    # Create validation datasets
    if hasattr(train_data, 'numpy'):
        val_data = tf.gather(train_data, val_idx)
        new_train_data = tf.gather(train_data, train_idx)
    else:
        val_data = train_data[val_idx]
        new_train_data = train_data[train_idx]
    
    if len(train_labels.shape) > 1:  # One-hot encoded
        val_labels = train_labels[val_idx]
        new_train_labels = train_labels[train_idx]
    else:
        val_labels = train_labels[val_idx]
        new_train_labels = train_labels[train_idx]
    
    return new_train_data, val_data, new_train_labels, val_labels

# Now modify the MLP class to include validation performance tracking
class MLP:
  def __init__(self, train_data, train_labels, hidden_layer=32, loss_fn='mse', optimizer='adam', epochs=10, batch_size=32):
    # initialise constructor with changeable labels
    self.train_data = train_data
    self.train_labels = train_labels
    self.hidden_layer = hidden_layer
    self.epochs = epochs
    self.batch_size = batch_size
    self.loss_fn = loss_fn
    self.optimizer = optimizer

    # Define our common structure (single hidden layer)
    self.mlp_model = Sequential()
    self.mlp_model.add(Dense(self.hidden_layer, input_shape=(self.train_data.shape[1],), activation='sigmoid'))
    self.mlp_model.add(Dense(len(np.unique(self.train_labels)), activation='softmax')) # output probability distribution for labels

    # Configure the model and start training
    self.mlp_model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=['accuracy'])
    
    # Create validation set
    train_data_split, val_data_split, train_labels_split, val_labels_split = create_validation_split(
        self.train_data, self.train_labels
    )
    
    # Train with explicit validation data
    self.history = self.mlp_model.fit(
        train_data_split, 
        train_labels_split, 
        epochs=self.epochs, 
        batch_size=self.batch_size, 
        verbose=0, 
        validation_data=(val_data_split, val_labels_split), 
        shuffle=True
    )
    
    # Store validation data for later use
    self.val_data = val_data_split
    self.val_labels = val_labels_split

  def perf(self, data, labels):
    # Test the model after training
    outputs = self.mlp_model.predict(data)
    outputs = np.round(outputs)
    # Return metrics for results
    return accuracy_score(outputs, labels), precision_score(outputs, labels, average='weighted'), recall_score(outputs, labels, average='weighted'), f1_score(outputs, labels, average='weighted')

# Modified CNN class with improved validation tracking
class CNN:
  def __init__(self, train_data, train_labels, loss_fn='mse', optimizer='adam', epochs=10, batch_size=32):
    # define constructor with changeable parameters
    self.train_data = train_data
    self.train_labels = train_labels
    self.epochs = epochs
    self.batch_size = batch_size
    self.loss_fn = loss_fn
    self.optimizer = optimizer

    # Configure network (Conv1D, Pooling, Flatten, FCL, FCL, Output)
    self.cnn_model = Sequential()
    self.cnn_model.add(Conv1D(64, kernel_size=8, strides=2, padding='same', activation='relu', input_shape=(self.train_data.shape[1], 1)))
    self.cnn_model.add(MaxPooling1D(2))
    self.cnn_model.add(Flatten())
    self.cnn_model.add(Dense(32, activation='relu'))
    self.cnn_model.add(Dense(16, activation='relu'))
    self.cnn_model.add(Dense(len(np.unique(self.train_labels)), activation='softmax'))

    # Create validation set
    train_data_split, val_data_split, train_labels_split, val_labels_split = create_validation_split(
        self.train_data, self.train_labels
    )
    
    # Need to reshape for CNN
    if hasattr(train_data_split, 'numpy'):
        train_data_reshaped = train_data_split.numpy().reshape(train_data_split.shape[0], train_data_split.shape[1], 1)
        val_data_reshaped = val_data_split.numpy().reshape(val_data_split.shape[0], val_data_split.shape[1], 1)
    else:
        train_data_reshaped = train_data_split.reshape(train_data_split.shape[0], train_data_split.shape[1], 1)
        val_data_reshaped = val_data_split.reshape(val_data_split.shape[0], val_data_split.shape[1], 1)

    # Set parameters and start training with explicit validation data
    self.cnn_model.compile(loss=self.loss_fn, optimizer=self.optimizer, metrics=['accuracy'])
    self.history = self.cnn_model.fit(
        train_data_reshaped, 
        train_labels_split, 
        epochs=self.epochs, 
        batch_size=self.batch_size, 
        shuffle=True, 
        validation_data=(val_data_reshaped, val_labels_split), 
        verbose=0
    )
    
    # Store validation data for later use
    self.val_data = val_data_reshaped
    self.val_labels = val_labels_split

  def perf(self, data, labels):
    # Test the model after training
    if hasattr(data, 'numpy'):
        data_reshaped = data.numpy().reshape(data.shape[0], data.shape[1], 1)
    else:
        data_reshaped = data.reshape(data.shape[0], data.shape[1], 1)
    
    outputs = np.round(self.cnn_model.predict(data_reshaped))
    # Return metrics for results
    return accuracy_score(outputs, labels), precision_score(outputs, labels, average='weighted'), recall_score(outputs, labels, average='weighted'), f1_score(outputs, labels, average='weighted')

# Modified DAE class with validation tracking
class DAE:
  def __init__(self, train_data, test_data, latent=64, epochs=10, batch_size=32, optimizer='adam', loss_fn='mse'):
    # initialise constructor and changeable parameters
    self.latent = latent # hidden layer in middle of network
    self.train_data = train_data
    self.test_data = test_data
    self.epochs = epochs
    self.batch_size = batch_size
    self.optimizer = optimizer
    self.loss_fn = loss_fn

    # Configure network
    self.dae_model = Sequential()
    
    self.dae_model.add(Dense(16, activation='relu'))
    self.dae_model.add(Dense(32, activation='relu'))
    self.dae_model.add(Dense(self.latent, activation='relu', input_shape=(self.train_data.shape[1],))) # latent space
    self.dae_model.add(Dense(32, activation='relu'))    
    self.dae_model.add(Dense(16, activation='relu'))
    self.dae_model.add(Dense(self.train_data.shape[1], activation='sigmoid')) 

    # Configure network and train
    self.dae_model.compile(optimizer=self.optimizer, loss=self.loss_fn)
    
    # Create validation set (10% of training data)
    train_data_split, val_data_split, _, _ = create_validation_split(
        self.train_data, np.zeros((len(self.train_data),1))  # Dummy labels since autoencoders don't use them
    )
    
    # Add Gaussian noise to training input
    noisy_train_data = train_data_split + tf.random.normal(shape=train_data_split.shape, mean=0.0, stddev=0.05)

    # Train with explicit validation data
    self.history = self.dae_model.fit(
        noisy_train_data, 
        train_data_split, 
        epochs=self.epochs, 
        batch_size=self.batch_size, 
        validation_data=(val_data_split, val_data_split), 
        shuffle=True, 
        verbose=0
    )
    
    # Store validation data
    self.val_data = val_data_split

  def perf(self, data, labels, standard_devs=3):
    # Find loss from reproducing normal ECGs after being trained on normal ECGs
    normal_reproduction_loss = tf.keras.losses.mse(self.dae_model.predict(self.train_data), self.train_data)
    
    # Find loss between reproduced images and original normal images
    self.error_threshold = np.mean(normal_reproduction_loss) + standard_devs*np.std(normal_reproduction_loss)

    # Reproduce test data and find loss between these and original test data
    test_reproduction = self.dae_model.predict(data)
    test_reproduction_loss = tf.keras.losses.mse(test_reproduction, data)

    predictions = []
    # if loss is greater than threshold from normal images, it is abnormal
    for i in range(len(test_reproduction_loss)):
      if np.mean(test_reproduction_loss[i]) > self.error_threshold:
        predictions.append([1,0])
      else:
        predictions.append([0,1])

    predictions = np.array(predictions)

    return accuracy_score(predictions, labels), precision_score(predictions, labels, average='weighted'), recall_score(predictions, labels, average='weighted'), f1_score(predictions, labels, average='weighted')

# Add this code after the original model training to print validation metrics
# Replace the original model training code with these updated versions

# Train and evaluate MLP
start = timer()
mlp_model = MLP(train_data, train_labels)
end = timer()
training_times.append(end-start)
metrics.append(mlp_model.perf(test_data, test_labels))

print(mlp_model.perf(test_data, test_labels))
print_validation_metrics(mlp_model, mlp_model.val_data, mlp_model.val_labels, "MLP")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(mlp_model.history.history['loss'], label="Training Loss", linewidth=3)
plt.plot(mlp_model.history.history['val_loss'], label="Validation Loss", linewidth=3)
plt.plot(mlp_model.history.history['accuracy'], label="Training Accuracy", linewidth=3, linestyle='--')
plt.plot(mlp_model.history.history['val_accuracy'], label="Validation Accuracy", linewidth=3, linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss / Accuracy')
plt.title('MLP Training Metrics')
plt.legend(prop={'size': 12})
plt.grid(True, alpha=0.3)
plt.show()

# Train and evaluate CNN
start = timer()
cnn_model = CNN(train_data, train_labels)
end = timer()
training_times.append(end-start)
metrics.append(cnn_model.perf(test_data, test_labels))

print(cnn_model.perf(test_data, test_labels))
print_validation_metrics(cnn_model, cnn_model.val_data, cnn_model.val_labels, "CNN")

# Plot CNN training and validation metrics with improved visualization
plt.figure(figsize=(12, 8))

# Create subplot for loss
plt.subplot(2, 1, 1)
epochs = np.arange(1, cnn_model.epochs + 1)
plt.plot(epochs, cnn_model.history.history['loss'], 'b-', label="Training Loss", linewidth=2)
plt.plot(epochs, cnn_model.history.history['val_loss'], 'r-', label="Validation Loss", linewidth=2)
plt.title("CNN Loss Trend", fontsize=14)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', prop={'size': 12})

# Create subplot for accuracy
plt.subplot(2, 1, 2)
plt.plot(epochs, cnn_model.history.history['accuracy'], 'b-', label="Training Accuracy", linewidth=2)
plt.plot(epochs, cnn_model.history.history['val_accuracy'], 'r-', label="Validation Accuracy", linewidth=2)
plt.title("CNN Accuracy Trend", fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='lower right', prop={'size': 12})

plt.tight_layout()
plt.show()

# Train and evaluate DAE
start = timer()
dae_model = DAE(normal_train_data, normal_test_data)
end = timer()
training_times.append(end-start)
metrics.append(dae_model.perf(test_data, test_labels))

print(dae_model.perf(test_data, test_labels))
print_validation_metrics(dae_model, test_data, test_labels, "DAE")

# Plot training and validation losses for DAE with improved visualization
plt.figure(figsize=(10, 6))
epochs = np.arange(1, dae_model.epochs + 1)
plt.plot(epochs, dae_model.history.history["loss"], 'b-', label="Training Loss", linewidth=3)
plt.plot(epochs, dae_model.history.history["val_loss"], 'r-', label="Validation Loss", linewidth=3)
plt.title("DAE Loss Trend", fontsize=14)
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(prop={'size': 12})
plt.show()

# Add a function to compare model performances
def compare_model_performances(metrics, models, training_times):
    """
    Compare performance metrics of different models
    """
    print("\n===== Model Performance Comparison =====")
    print(f"{'Model':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10} {'Train Time (s)':<15}")
    print("-" * 65)
    
    for i, model_name in enumerate(models):
        if i < len(metrics):
            acc, prec, rec, f1 = metrics[i]
            time = training_times[i]
            print(f"{model_name:<10} {acc:.4f}     {prec:.4f}     {rec:.4f}     {f1:.4f}     {time:.2f}")

# Call the comparison function after all models are trained
compare_model_performances(metrics, cls, training_times)