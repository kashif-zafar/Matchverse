import pandas as pd
import xgboost as xgb
import random
import os
from sklearn.preprocessing import LabelEncoder


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'models')


os.makedirs(MODEL_DIR, exist_ok=True)

print("⏳ Loading Data...")
users_path = os.path.join(DATA_DIR, 'users.csv')
interactions_path = os.path.join(DATA_DIR, 'interactions.csv')

users_df = pd.read_csv(users_path)
interactions_df = pd.read_csv(interactions_path)


label_encoders = {}
for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
    le = LabelEncoder()
    users_df[col] = le.fit_transform(users_df[col])
    label_encoders[col] = le


interaction_counts = interactions_df["Target_ID"].value_counts()

print("⚙️ Generating Training Data...")


train_data = []

existing_pairs = set(zip(interactions_df["Member_ID"], interactions_df["Target_ID"]))

for _, row in interactions_df.iterrows():
    train_data.append([row["Member_ID"], row["Target_ID"], 1])  


all_member_ids = users_df["Member_ID"].unique()
user_lookup = users_df.set_index("Member_ID")

num_negatives = len(train_data)
count = 0

while count < num_negatives:
    u1 = random.choice(all_member_ids)
    u2 = random.choice(all_member_ids)
    
    # Logic: Must be opposite gender & not already interacted
    if user_lookup.loc[u1]["Gender"] != user_lookup.loc[u2]["Gender"]:
        if (u1, u2) not in existing_pairs:
            train_data.append([u1, u2, 0])  # 0 = No Match
            count += 1

# Convert to DataFrame
train_df = pd.DataFrame(train_data, columns=["Member_ID", "Target_ID", "Interaction"])



print("Feature Engineering...")

# Merge user details for both Initiator (Member) and Target
train_df = train_df.merge(users_df, left_on="Member_ID", right_on="Member_ID", suffixes=('_m', '_t'))
train_df = train_df.merge(users_df, left_on="Target_ID", right_on="Member_ID", suffixes=('', '_target')) 


def calculate_features(row):
    # Get user details from the lookup we created earlier
    u1 = user_lookup.loc[row["Member_ID"]]
    u2 = user_lookup.loc[row["Target_ID"]]
    
    return pd.Series({
        "Age_Diff": abs(u1["Age"] - u2["Age"]),
        "Same_Caste": 1 if u1["Caste"] == u2["Caste"] else 0,
        "Same_Sect": 1 if u1["Sect"] == u2["Sect"] else 0,
        "Same_State": 1 if u1["State"] == u2["State"] else 0,
        "Target_Popularity": interaction_counts.get(row["Target_ID"], 0)
    })

# Apply feature calculation
features_df = train_df.apply(calculate_features, axis=1)
X = features_df
y = train_df["Interaction"]


print("🚀 Training XGBoost Model...")

model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric="logloss",
    use_label_encoder=False
)

model.fit(X, y)

output_path = os.path.join(MODEL_DIR, 'recommendation_model.json')
model.save_model(output_path)

print(f"✅ Success! XGBoost model saved to: {output_path}")
print(f"   - Training Accuracy: {model.score(X, y):.4f}")