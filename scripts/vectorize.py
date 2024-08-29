from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir, val_dir, test_dir, batch_size, img_size):
    train_datagen = ImageDataGenerator(rescale = 1./255)
validation_datagen = ImageDataGenerator(rescale = 1./255)

# Train Generator
train_generator = train_datagen.flow_from_directory(

    train_dir,
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)

# Validation Generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

return train_generator, validation_generator
