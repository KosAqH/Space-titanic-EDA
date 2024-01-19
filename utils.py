import os
import pandas as pd
import numpy as np


def fill_na_with_range(df, src_col, src_cat_val, col_to_fill):
    min_value = df.loc[df[src_col]==src_cat_val, col_to_fill].min()
    max_value = df.loc[df[src_col]==src_cat_val, col_to_fill].max()

    n_empty = df[df[src_col] == src_cat_val][col_to_fill].isna().sum()

    random_gen = np.random.default_rng()
    random_vals = random_gen.integers(min_value, max_value, size=n_empty)

    df.loc[(df[src_col].dropna() == src_cat_val) & (df[col_to_fill].isna()), col_to_fill] = random_vals

def fill_na_with_sample(df, src_col, src_cat_val, col_to_fill):
    for_sample = df[df[src_col] == src_cat_val][col_to_fill].value_counts(normalize=True)
    n_empty = df[df[src_col] == src_cat_val][col_to_fill].isna().sum()

    random_gen = np.random.default_rng()
    random_vals = random_gen.choice(for_sample.index.to_list(),
                           size=n_empty,
                           p=for_sample.values.tolist())
    df.loc[(df[src_col].dropna() == src_cat_val) & (df[col_to_fill].isna()), col_to_fill] = random_vals   

def data_transform_pipeline(
        filename: str,
        out_filename: str = "test",
        path: str = "data"
    ) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(path, filename))

    ## Create new columns
    # Tickets
    df[["TicketId", "InvidualId"]] = df["PassengerId"].str.split("_", expand=True)
    
    # Names
    df[["FirstName", "LastName"]] = df["Name"].str.split(" ", expand=True)
    df = df.drop(columns=["FirstName", "Name"])

    # Cabins
    df[["Deck", "Num", "Side"]] = df["Cabin"].str.split("/", expand=True)
    df["Num"] = df["Num"].astype('Int16')
    df["Num_bucket"] = df["Num"].apply(lambda x: x // 100 * 100) 
    df = df.drop(columns=["Cabin"])

    # Spendings
    expenses_columns = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    df["Total spendings"] = df[expenses_columns].sum(axis=1)

    ## Impute missing values
    # Impute home planet
    fill_na_with_sample(df, "Destination", "55 Cancri e", "HomePlanet")
    fill_na_with_sample(df, "Destination", "PSO J318.5-22", "HomePlanet")
    fill_na_with_sample(df, "Destination", "TRAPPIST-1e", "HomePlanet")

    for deck in df["Deck"].dropna().unique():
        fill_na_with_sample(df, "Deck", deck, "HomePlanet")
    df["HomePlanet"] = df["HomePlanet"].fillna(df["HomePlanet"].mode())

    # Impute age
    df["Age_bucket"] = df["Age"].apply(lambda x: x // 10 * 10) 
    mean_age = df.groupby("Deck")["Age_bucket"].mean().apply(lambda x: round(x))
    for deck in df["Deck"].dropna().unique():
        df.loc[(df["Age"].isna()) & (df["Deck"] == deck), "Age"] = mean_age[deck] 
    df["Age"] = df["Age"].fillna(df["Age"].mean())
    df["Age_bucket"] = df["Age"].apply(lambda x: x // 10 * 10) 

    # Impute VIP
    n_to_set_true = int(df["VIP"].isna().sum() * 0.025)
    idx_to_set_true = df.loc[df["VIP"].isna(), "Total spendings"].sort_values(ascending=False).head(n_to_set_true).index
    df.loc[idx_to_set_true, "VIP"] = True
    df["VIP"] = df["VIP"].fillna(False)

    # Impute Cryosleep
    df.loc[(df["CryoSleep"].isna()) & (df["Total spendings"] == 0), "CryoSleep"] = False
    df.loc[(df["CryoSleep"].isna()) & (df["Total spendings"] > 0), "CryoSleep"] = True

    df.loc[(df["CryoSleep"] == True), expenses_columns] = 0
    df["Total spendings"] = df[expenses_columns].sum(axis=1)

    # Impute deck
    fill_na_with_sample(df, "HomePlanet", "Earth", "Deck")
    fill_na_with_sample(df, "HomePlanet", "Mars", "Deck")
    fill_na_with_sample(df, "HomePlanet", "Europa", "Deck")

    # Impute side
    for deck in df["Deck"].dropna().unique():
        fill_na_with_sample(df, "Deck", deck, "Side")

    # Impute num
    for deck in df["Deck"].dropna().unique():
        fill_na_with_range(df, "Deck", deck, "Num")
    df["Num_bucket"] = df["Num"].apply(lambda x: x // 100 * 100) 

    # Impute destination
    fill_na_with_sample(df, "HomePlanet", "Earth", "Destination")
    fill_na_with_sample(df, "HomePlanet", "Mars", "Destination")
    fill_na_with_sample(df, "HomePlanet", "Europa", "Destination")

    # Impute expenses
    for col in expenses_columns:
        df.loc[(df[col].isna()) & (df["CryoSleep"] == True), col] = 0
        df[col] = df.groupby(["VIP", "Age_bucket"])[col].transform(lambda x: x.fillna(x.mean()))

    # Create columns
    # Is_travelling_in_group
    df.loc[:, "is_travelling_in_group"] = False
    df.loc[(df.duplicated("TicketId", keep=False)), "is_travelling_in_group"] = True

    print(df.isna().sum())
    df = df.fillna(0)

    # Encode non-numeric data
    df["TicketId"] = df["TicketId"].astype(int)
    df["InvidualId"] = df["InvidualId"].astype(int)
    df["CryoSleep"] = df["CryoSleep"].astype(int)
    df["VIP"] = df["VIP"].astype(int)

    deck_values, codes = df["Deck"].factorize(sort=True)
    df["Deck"] = deck_values
    deck_values, codes = df["Side"].factorize(sort=True)
    df["Side"] = deck_values

    df["HomePlanet_Orig"] = df["HomePlanet"]
    df["Destination_Orig"] = df["Destination"]

    df = pd.get_dummies(df, columns=['HomePlanet', "Destination"])

    df.to_csv(os.path.join("data", f"{out_filename}_transformed.csv"), index=False)