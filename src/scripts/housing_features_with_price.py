import pandas as pd
df = pd.read_csv("bamboo_full_raw_data.csv")
feat = pd.read_csv("Housing_Features.csv")
print(df.shape[0])
print(feat.shape[0])
feat['Price'] = df['Price']
feat.to_csv("Housing_Features_with_Price.csv")