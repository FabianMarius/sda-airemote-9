import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path


DATA_ROOT = Path("E:\\SDA\\Courses\\AIRemote\\Repos\\sda-airemote-9-course-materials\\course_7\\AIEngineerModelPackaging\\data\\")


def get_splits():
    df_rbb = pd.read_csv(DATA_ROOT / "processed" / "df_processed.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df_rbb.drop(columns=['price']),
        df_rbb['price'],
        test_size=0.2,
        random_state=42
    )
    return  X_train, X_test, y_train, y_test


def train_simple_lr():

    X_train, X_test, y_train, y_test = get_splits()

    model_lr = LinearRegression()
    model_lr.fit(X_train, y_train)
    print(model_lr.score(X_train, y_train))
    print(model_lr.score(X_test, y_test))


def train_scaled_lr():
    X_train, X_test, y_train, y_test = get_splits()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model_lr = LinearRegression()
    model_lr.fit(X_train_scaled, y_train)
    print(model_lr.score(X_train_scaled, y_train))
    print(model_lr.score(X_test_scaled, y_test))


def train_model(model_type):
    if model_type == "lr":
        train_simple_lr()
    elif model_type == "scaled_lr":
        train_scaled_lr()


if __name__ == "__main__":
    train_model("scaled_lr")
    