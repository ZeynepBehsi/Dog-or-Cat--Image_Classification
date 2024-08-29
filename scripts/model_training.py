from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


def create_vgg16_model(input_shape):
    # Load the pre-trained VGG16 model
    conv_model = VGG16(weights="imagenet", 
                       include_top=False,  # Don't load the last layer (classification) of model
                       input_shape=input_shape)

    # Freeze the layers of VGG16 model to prevent training
    conv_model.trainable = False

    # Build Model
    model = Sequential([
        conv_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),  # Add dropout: randomly disable 50% neurons in each epoch, to avoid overfitting
        Dense(1, activation="sigmoid")
    ])

    # Compile the model with Adam optimizer and binary cross-entropy loss
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(model, train_generator, val_generator, epochs):
    # Stop training when a monitored quantity has stopped improving
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Train the model with the specified callbacks
    history = model.fit(train_generator, 
                        validation_data=val_generator, 
                        epochs=epochs, 
                        callbacks=[early_stopping, reduce_lr])
    
    return history
