class CyclicOrdinalRegressor:
    
    def __init__(self, numerical_vars, categorical_vars, name):
        self.categorical_vars = categorical_vars
        self.numerical_vars = numerical_vars
        self.model_name = name + '.keras'
        self.embedding_layers = []
        self.input_layers = []
        self.embedding_sizes = {}
        self.model = None
        self.label_encoders = {}
        self.max_categorical_values = {}  # NEW: Track max values
        
    def calculate_embedding_sizes(self, data):
        """Calculate embedding dimensions with buffer"""
        for col in self.categorical_vars:
            # Find maximum value across all data
            self.max_categorical_values[col] = int(data[col].max())  # NEW
            
            # Calculate embedding size with 20% buffer
            unique_vals = len(data[col].unique())
            self.embedding_sizes[col] = max(2, min(50, round(1.6 * np.log2(unique_vals))))
            
            print(f"{col}: Max value={self.max_categorical_values[col]}, Embedding size={self.embedding_sizes[col]}")

    def _build_categorical_inputs(self):
        """Create embedding layers with safe dimensions"""
        for col in self.categorical_vars:
            input_layer = Input(shape=(1,), name=col)
            
            # NEW: Use max_value + buffer for embedding dimension
            embedding_dim = self.max_categorical_values[col] + 1000  # Safety buffer
            embedding = Embedding(
                input_dim=embedding_dim,  # Now large enough
                output_dim=self.embedding_sizes[col],
                name=f'embed_{col}'
            )(input_layer)
            flattened = Flatten()(embedding)
            
            self.input_layers.append(input_layer)
            self.embedding_layers.append(flattened)

    def preprocess_data(self, data, is_train=True):
        """Handle unseen categories safely"""
        X_categorical = []
        
        for col in self.categorical_vars:
            if is_train:
                # NEW: No label encoding needed for numeric IDs
                # Just ensure they're integers and track max
                data[col] = data[col].astype(int)
                X_categorical.append(data[col].values)
            else:
                # NEW: Clip to max seen during training
                data[col] = np.minimum(
                    data[col].astype(int),
                    self.max_categorical_values[col]
                )
                X_categorical.append(data[col].values)
        
        X_numerical = data[self.numerical_vars].values
        
        return X_categorical + [X_numerical]

    # ... [rest of the class remains the same] ...
