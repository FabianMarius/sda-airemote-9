import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

cols_to_drop = [
    'id',
    'name',
    # 'rating',
    # 'reviews',
    'host_name',
    'host_id',
    # 'address',
    'features',
    'amenities',
    'safety_rules',
    'hourse_rules',
    'img_links',
    # 'price',
    'country',
    # 'bathrooms',
    # 'beds',
    # 'guests',
    # 'toiles',
    # 'bedrooms',
    # 'studios',
    # 'checkin',
    # 'checkout'
]

DATA_ROOT = Path("E:\\SDA\\Courses\\AIRemote\\Repos\\sda-airemote-9-course-materials\\course_7\\AIEngineerModelPackaging\\data\\")


def add_host_count(df):  # transmitere parametru prin referinta
    df_host_counts = df.groupby('host_id')['name'].nunique().rename(
        'host_count'
    )
    df['host_count'] = df_host_counts.loc[df['host_id']].values


def add_city_info(df):
    df['address_0'] = df['address'].str.split(',').str[0]
    df_address_0_counter = df['address_0'].value_counts().loc[df['address_0']]
    df['address_0'] = np.where(df_address_0_counter > 50, df['address_0'], 'other')


def process_rating(df):
    df_rating = np.where(
        df['rating'] == 'New',
        np.nan,
        df['rating']
    )

    df_rbb_rating_mean = pd.Series(df_rating.astype(float)).dropna().mean()
    df_rating = np.where(
        pd.Series(df_rating).isna(),
        df_rbb_rating_mean,
        df['rating']
    ).astype(float)

    df['rating'] = df_rating


def process_reviews(df):
    df['reviews'] = df['reviews'].str.replace(',', '').astype(int)


def add_one_hot_encoded_city_info(df):
    encoder = OneHotEncoder(sparse_output=False)
    df_address_0_ohe = encoder.fit_transform(df[['address_0']])

    df_address_0_ohe = pd.DataFrame(
        df_address_0_ohe,
        columns=encoder.get_feature_names_out()
    )

    # returneaza un nou obiect, care nu este acelasi cu cel returnat ca parametru
    df = pd.concat([df, df_address_0_ohe], axis=1)
    return df
    

def preprocess(input_filepath, output_filepath):
    df_rbb = pd.read_csv(input_filepath, index_col=0)
    add_host_count(df_rbb)
    
    #######
    df_rbb = df_rbb.drop(columns=cols_to_drop)
    #######

    add_city_info(df_rbb)

    #######
    df_rbb = df_rbb.drop(columns=['address', 'checkout', 'checkin'])
    #######

    process_rating(df_rbb)
    process_reviews(df_rbb)
    df_rbb = add_one_hot_encoded_city_info(df_rbb)

    df_rbb = df_rbb.drop(columns=['address_0'])

    df_rbb.to_csv(output_filepath, index=None)



if __name__ == "__main__":
    input_file = DATA_ROOT / "raw" / "airbnb.csv.zip"
    output_file = DATA_ROOT / "processed" / "df_processed.csv"
    preprocess(input_file, output_file)
