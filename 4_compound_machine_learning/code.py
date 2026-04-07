from typing import List, Union
import numpy.typing as npt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from sklearn.model_selection import GridSearchCV

def draw_molecule(csvfile: str) -> None:
    # 課題 4-1
    csv = pd.read_csv(csvfile)
    row  = csv[csv["Compound ID"] == "CHEMBL540227"]
    smile = row["SMILES"].iat[0]
    mol = Chem.MolFromSmiles(smile)
    Draw.MolToFile(mol, "data/CHEMBL540227.png")
    pass

def create_2d_descriptors(smiles: str) -> Union[npt.NDArray[np.float64], List[float]]:
    # 課題 4-2
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    desc_values = []
    for _, desc_func in Descriptors._descList:
        desc_values.append(desc_func(mol))
    return desc_values



def predict_logpapp(csvfile: str) -> Union[npt.NDArray[np.float64], pd.Series, List[float]]:
    # 課題 4-3
    np.random.seed(0) # 出力を固定するためにseedを指定
    rfr = RandomForestRegressor(random_state=0) # 出力を固定するためにrandom_stateを指定

    x = []
    y = []
    csv = pd.read_csv(csvfile)
    for _, row in csv.iterrows():
        smiles = str(row["SMILES"]).strip()
        desc_values = create_2d_descriptors(smiles)
        if len(desc_values) == 0:
            continue
        x.append(desc_values)
        y.append(row["LogP app"])

    X = np.array(x, dtype=np.float64)
    Y = np.array(y, dtype=np.float64)
    # Xは説明変数、yは目的変数
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=700, random_state=0)
    rfr.fit(X_train, y_train)
    Y_pred = rfr.predict(X_test)

    return Y_pred

def grid_search(csvfile: str) -> float:
    # 課題 4-4
    # こちらも出力を固定するためにseedやrandom_stateを指定すること
    np.random.seed(0)

    x = []
    y = []
    csv = pd.read_csv(csvfile)
    for _, row in csv.iterrows():
        smiles = str(row["SMILES"]).strip()
        desc_values = create_2d_descriptors(smiles)
        if len(desc_values) == 0:
            continue
        x.append(desc_values)
        y.append(row["LogP app"])

    X = np.array(x, dtype=np.float64)
    Y = np.array(y, dtype=np.float64)
    # Xは説明変数、yは目的変数
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=700, random_state=0)

    rfr = RandomForestRegressor(random_state=0)  # 出力を固定するためにrandom_stateを指定
    param_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [5, 10, 15],
    }

    gscv = GridSearchCV(
        estimator=rfr,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=4,
    )

    gscv.fit(X_train, y_train)

    best_model = gscv.best_estimator_
    Y_pred = best_model.predict(X_test)
    RMSE = mean_squared_error(y_test, Y_pred, squared=False)

    return RMSE

if __name__ == "__main__":
    smiles = "C(=O)(c1ccc(OCCCCCC)cc1)CCNc1cc(Cl)ccc1"
    filepath = "data/fukunishi_data.csv"
    # 課題 4-1
    draw_molecule(filepath)
    # 課題 4-2
    print(create_2d_descriptors(smiles))
    # 課題 4-3
    print(predict_logpapp(filepath))
    # 課題 4-4
    print(grid_search(filepath))
