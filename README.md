

class ClassifierModel:
    
    def __init__(self, numerical_vars, categorical_vars, name):
        self.categorical_vars = categorical_vars
        self.numerical_vars = numerical_vars
        self.model_name = name+'.keras'
        self.embedding_layers = []
        self.input_layers = []
        self.embedding_sizes = {}
        self.normalization_layer = None
        self.model = None
        self.categorical_inputs=[]
        self.num_enc = {}
        self.label_encoders={}

    def calculate_embedding_sizes(self, data):
        for col in self.categorical_vars:
            unique_vals = len(data[col].unique())
            self.embedding_sizes[col] = max(2, round(np.log2(unique_vals)))

    def preprocess_categorical_inputs(self, data):
        categorical_inputs=[]
        for col in self.categorical_vars:
            input_layer = Input(shape=(1,), name=col)
            embedding_layer = Embedding(input_dim=len(data[col].unique())+1, output_dim=self.embedding_sizes[col])(input_layer)
            self.embedding_layers.append(embedding_layer)
            self.categorical_inputs.append(Flatten()(embedding_layer))
            self.input_layers.append(input_layer)
            


    def build_model(self, num_size):

        # Numerical inputs (assuming you have separate numerical features)
        numerical_input = Input(shape=(num_size,), name='numerical_input')
        concatenated_inputs = Concatenate()(self.categorical_inputs + [numerical_input])
        hidden_layer = Dense(64, activation='relu')(concatenated_inputs)
#         batch_norm_layer = BatchNormalization()(hidden_layer)
#         dropout_layer = Dropout(rate=dropout_rate)(batch_norm_layer)
        hidden_layer = Dense(32, activation='relu')(hidden_layer)
#         batch_norm_layer = BatchNormalization()(hidden_layer)
#         dropout_layer = Dropout(rate=0.2)(batch_norm_layer)
        hidden_layer = Dense(32, activation='relu')(hidden_layer)
#         batch_norm_layer = BatchNormalization()(hidden_layer)
#         dropout_layer = Dropout(rate=0.1)(batch_norm_layer)
        hidden_layer = Dense(16, activation='relu')(hidden_layer)
#         batch_norm_layer = BatchNormalization()(hidden_layer)
#         dropout_layer = Dropout(rate=0.05)(batch_norm_layer)
        output_layer = Dense(1, activation='sigmoid')(hidden_layer)

        self.model = Model(inputs=self.input_layers + [numerical_input], outputs=output_layer)
        print(self.model.summary())
        
        
    def preprocess_data(self, data , is_train=True):
        # Convert text categories to numerical representations using LabelEncoder
        
        X_categorical = [] 
        
        for col in self.categorical_vars:
            print(col)
            if is_train:
                encoder = LabelEncoder()
                S = data[col]
                S=pd.concat([S,pd.Series(['UNK'])])
                S=S.astype('str').astype('category')
                data[col] = data[col].astype('str').astype('category')
                encoder.fit(S)
                X_categorical.append(encoder.transform(data[col]))
                self.label_encoders[col] = encoder
            else:
                
                encoder = self.label_encoders.get(col)
               
                if encoder:
                    
                    rep = { x: 'UNK' for x in data[col].unique() if x not in encoder.classes_ } # Ordinal Encoder
                    
                    data[col] = data[col].replace(rep)
                    
                    data[col] = data[col].astype('str').astype('category')
                    
                    X_categorical.append(encoder.transform(data[col]))
                
                else:
                    
                    raise ValueError(f"No encoder found for column {col}")
                    
                    
        X_numerical = data[self.numerical_vars]
        
#         if is_train:
            
#             self.num_enc['mean'] = np.mean(X_numerical, axis=0)
            
#             self.num_enc['std'] = np.std(X_numerical, axis=0)
            
#             X_numerical = (X_numerical - self.num_enc['mean']) / self.num_enc['std']
            
#         else:
            
#             if self.num_enc.get('mean') is not None:
                
#                 X_numerical = (X_numerical - self.num_enc['mean']) / self.num_enc['std']
                
#             else:
                
#                 raise ValueError(f"No encoder found for numeric columns")

        # Manual standardization for numerical data

        return X_categorical + [X_numerical.to_numpy()]




    def compile_model(self):
        
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
        
#         plot_model(self.model, to_file=f'{self.model_name[:-6]}.png', show_shapes=True)

    def train_model(self, X , target_column, epochs=100, batch_size=32768, validation_split=0.05):
        history = self.model.fit(x=X,
                       y=target_column,
                       epochs=epochs,
                       batch_size=batch_size,
                       validation_split=validation_split)

        plt.plot(history.history['auc'])
        plt.plot(history.history['val_auc'])
        plt.title('Model AUC')
        plt.ylabel('AUC')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f'Graphs/{self.model_name[:-6]} AUC.jpg')
        plt.show()        
        
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f'Graphs/{self.model_name[:-6]} Loss.jpg')
        plt.show()        
        self.model.save(self.model_name)

