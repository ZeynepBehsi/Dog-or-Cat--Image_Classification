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
