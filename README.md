def preprocess_data(self, data, is_train=True):
        """Prepare all input types with proper transformations"""
        X_categorical = []
        
        # 1. Process categorical columns
        for col in self.categorical_vars:
            if is_train:
                le = LabelEncoder()
                data[f'{col}_enc'] = le.fit_transform(data[col])
                self.label_encoders[col] = le
            else:
                # le = self.label_encoders[col]
                # # Handle unseen categories by assigning to a new category
                # data[f'{col}_enc'] = data[col].apply(
                #     lambda x: le.transform([x])[0] if x in le.classes_ else self.max_categorical_values[col] + 1
                # )
                le = self.label_encoders[col]
                classes = set(le.classes_)
                max_val = self.max_categorical_values[col] + 1
                mask = data[col].isin(classes)

                data[f'{col}_enc'] = np.where(mask, le.transform(data[col][mask]), max_val)
                
            print(col)
            X_categorical.append(data[f'{col}_enc'].values.reshape(-1, 1))
        
        # 2. Process cyclical numerical features 
        X_numerical_cyclical = data[self.numerical_cyclical_vars].values
        
        # 3. Scale numerical non-cyclic features 
        if is_train:
            X_numerical_scaled = self.scaler.fit_transform(data[self.numerical_vars])
        else:
            X_numerical_scaled = self.scaler.transform(data[self.numerical_vars])
        
