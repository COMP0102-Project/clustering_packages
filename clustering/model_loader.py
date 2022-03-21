from keras.applications.densenet import DenseNet121
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout


class ChexpertModelLoader:
    def __init__(self, path_to_pretrained_chexpert_model, height, width, channels):
        self.PATH = path_to_pretrained_chexpert_model
        self.height = height
        self.width = width
        self.channels = channels

    def get_model(self):
        '''
        returns the output just before the fully connected layer of the chexpert model loaded
        through path_to_pretrained_chexpert_model
        :return:
        '''
        model = self.DenseNet1(self.height, self.width, self.channels)
        return self._get_penultimate_layer_model(model)

    def _get_penultimate_layer_model(self, model):
        x = model.input
        y = model.layers[-4].output
        return Model(x, y)

    def DenseNet1(self, height, width, channels):
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(height, width, channels))
        for layer in base_model.layers[-4:]:
            layer.trainable = False
        x = base_model.output
        x = GlobalAveragePooling2D(input_shape=(1024, 1, 1))(x)
        # Add a flattern layer
        x = Dense(2048, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        # Add a fully-connected layer
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        # and a logistic layer --  we have 5 classes
        predictions = Dense(6, activation='sigmoid')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        try:
            model.load_weights(self.PATH)
            print('Weights successfuly loaded')
        except:
            print('Weights not loaded')
        return model
