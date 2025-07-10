import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


class PreProcess:
    def __init__(self, input_data_path):
        self.class_names = None
        self.df=None
        self.preprocess_data(input_data_path)

   
    def preprocess_data(self,input_data):
        maths_dataset_df=pd.read_csv(input_data)
        # print(maths_dataset_df.head())
        # Math problems with type mentioned
        df_math = maths_dataset_df[maths_dataset_df['source'] == 'MATH'].copy()
        # Math problems without type mentioned
        df_gsm8k = maths_dataset_df[maths_dataset_df['source'] == 'GSM8K'].copy()
        pd.set_option('display.max_columns', None)
        # print('printing df_math with types',df_math.head(2))
        # print('printing df_gsm8k without types',df_gsm8k.head(2))
        df_math['type'] = df_math['type'].replace({
            'Algebra': 'Algebra',
            'Intermediate Algebra': 'Algebra',
            'Prealgebra': 'Algebra'
        })
        df_math['type'].value_counts()
        class_map = {
            'Algebra': 0,
            'Number Theory': 1,
            'Geometry': 2,
            'Precalculus': 3,
            'Counting & Probability': 4
        }

        # Create new column
        df_math['label'] = df_math['type'].map(class_map)
        self.df=df_math
        class_names = [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])]
        self.class_names=class_names

    def train_test_split(self,data):
        df_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)
        X_train, X_test, y_train, y_test = train_test_split(df_shuffled['problem'],df_shuffled['label'], stratify=df_shuffled['label'])
        return X_train, X_test, y_train, y_test
    

    def balanced_class_weights(self,y_train):
       class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
        )
       weights_dict = dict(zip(np.unique(y_train),class_weights))
       return weights_dict
    
        
if __name__ == "__main__":
    file_path="data/Math_problems.csv"
    p=PreProcess(file_path)
    class_names=p.class_names
    print(class_names)
    X_train, X_test, y_train, y_test = p.train_test_split(p.df)
    print(X_train.head(10),len(X_train))
    print(y_train.head(10),len(y_train))
    weights_dict=p.balanced_class_weights(y_train)
    print(weights_dict)

