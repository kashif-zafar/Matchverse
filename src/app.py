import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Page Config
st.set_page_config(page_title="MatchVerse - AI Matchmaking", layout="wide")
st.title("🔍 MatchVerse - AI Matchmaking Recommendations")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


USERS_CSV_PATH = os.path.join(BASE_DIR, '..', 'data', 'users.csv')
INTERACTIONS_CSV_PATH = os.path.join(BASE_DIR, '..', 'data', 'interactions.csv')
MODEL_JSON_PATH = os.path.join(BASE_DIR, '..', 'models', 'recommendation_model.json')

@st.cache_data
def load_users():
    if not os.path.exists(USERS_CSV_PATH):
        st.error(f"❌ File not found: {USERS_CSV_PATH}")
        return pd.DataFrame()
    return pd.read_csv(USERS_CSV_PATH)

@st.cache_data
def load_interactions():
    if not os.path.exists(INTERACTIONS_CSV_PATH):
        st.error(f"❌ File not found: {INTERACTIONS_CSV_PATH}")
        return pd.DataFrame()
    return pd.read_csv(INTERACTIONS_CSV_PATH)

users_df = load_users()
interactions_df = load_interactions()

# Stop execution if data didn't load
if users_df.empty or interactions_df.empty:
    st.stop()

# Label Encoding for categorical features
# NOTE: In a professional app, you would load saved encoders. 
# For now, we fit them fresh. Ideally, ensure users.csv hasn't changed sort order.
label_encoders = {}
for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
    le = LabelEncoder()
    users_df[col] = le.fit_transform(users_df[col])
    label_encoders[col] = le

# Load trained XGBoost model
bst = xgb.Booster()
if os.path.exists(MODEL_JSON_PATH):
    bst.load_model(MODEL_JSON_PATH)
else:
    st.error(f"❌ Model file not found at: {MODEL_JSON_PATH}")
    st.stop()

# Features used in the model
MODEL_FEATURES = ["Age_Diff", "Same_Caste", "Same_Sect", "Same_State", "Target_Popularity"]

# Precompute Target Popularity
interaction_counts = interactions_df["Target_ID"].value_counts()

def get_recommendations(member_id):
    """Fetch recommendations based on the trained model."""
    user_row = users_df[users_df["Member_ID"] == member_id]
    if user_row.empty:
        return {"error": "User not found"}

    user_meta = user_row.iloc[0]
    
    # Safe inverse transform (handling potential errors if encoding shifted)
    try:
        user_details = {
            "Member_ID": int(user_meta["Member_ID"]),
            "Gender": label_encoders["Gender"].inverse_transform([int(user_meta["Gender"])])[0],
            "Age": int(user_meta["Age"]),
            "Marital_Status": label_encoders["Marital_Status"].inverse_transform([int(user_meta["Marital_Status"])])[0],
            "Sect": label_encoders["Sect"].inverse_transform([int(user_meta["Sect"])])[0],
            "Caste": label_encoders["Caste"].inverse_transform([int(user_meta["Caste"])])[0],
            "State": label_encoders["State"].inverse_transform([int(user_meta["State"])])[0],
        }
    except Exception as e:
        return {"error": f"Error decoding user details: {str(e)}"}

    # Get opposite gender
    opposite_gender_encoded = 1 - user_meta["Gender"]
    eligible_profiles = users_df[users_df["Gender"] == opposite_gender_encoded].copy()

    # Exclude past interactions
    interacted_users = set(interactions_df[interactions_df["Member_ID"] == member_id]["Target_ID"])
    fresh_profiles = eligible_profiles[~eligible_profiles["Member_ID"].isin(interacted_users)].copy()

    if fresh_profiles.empty:
        return {"user_details": user_details, "recommended_profiles": [], "statistics": {}}

    # Feature Engineering
    fresh_profiles["Age_Diff"] = abs(fresh_profiles["Age"] - user_meta["Age"])
    fresh_profiles["Same_Caste"] = (fresh_profiles["Caste"] == user_meta["Caste"]).astype(int)
    fresh_profiles["Same_Sect"] = (fresh_profiles["Sect"] == user_meta["Sect"]).astype(int)
    fresh_profiles["Same_State"] = (fresh_profiles["State"] == user_meta["State"]).astype(int)
    fresh_profiles["Target_Popularity"] = fresh_profiles["Member_ID"].map(interaction_counts).fillna(0)

    # Prepare data for model
    X_test = fresh_profiles[MODEL_FEATURES]
    dtest = xgb.DMatrix(X_test)

    # Predict scores
    fresh_profiles["Score"] = bst.predict(dtest)

    # Get Top 100 recommendations
    recommended_profiles_df = fresh_profiles.sort_values(by="Score", ascending=False).head(100)

    # Decode categorical features
    for col in ["Gender", "Marital_Status", "Sect", "Caste", "State"]:
        recommended_profiles_df[col] = label_encoders[col].inverse_transform(recommended_profiles_df[col].astype(int))

    # Statistics
    statistics = {
        "age_distribution": dict(Counter(recommended_profiles_df["Age"])),
        "state_distribution": dict(Counter(recommended_profiles_df["State"])),
        "caste_distribution": dict(Counter(recommended_profiles_df["Caste"])),
    }

    return {
        "user_details": user_details,
        "recommended_profiles": recommended_profiles_df.to_dict(orient="records"),
        "statistics": statistics,
    }

# Streamlit UI
member_id = st.text_input("Enter Member ID:", "")

if st.button("Get Recommendations"):
    if member_id.isdigit():
        member_id = int(member_id)
        result = get_recommendations(member_id)
        
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("📌 User Details")
            st.json(result["user_details"])

            # Show recommended profiles
            st.subheader("🎯 Recommended Profiles")
            recommendations_df = pd.DataFrame(result["recommended_profiles"])
            if not recommendations_df.empty:
                st.dataframe(recommendations_df)

                # 📊 Charts
                st.subheader("📊 Statistics")
                
                # Check if we have data before plotting
                if result["statistics"]["age_distribution"]:
                    age_df = pd.DataFrame(result["statistics"]["age_distribution"].items(), columns=["Age", "Count"])
                    fig_age = px.bar(age_df, x="Age", y="Count", title="Age Distribution", color="Count", color_continuous_scale="Blues")
                    st.plotly_chart(fig_age, use_container_width=True)

                if result["statistics"]["caste_distribution"]:
                    caste_df = pd.DataFrame(result["statistics"]["caste_distribution"].items(), columns=["Caste", "Count"])
                    fig_caste = px.bar(caste_df, x="Caste", y="Count", title="Caste Distribution", color="Count", color_continuous_scale="Reds")
                    st.plotly_chart(fig_caste, use_container_width=True)

                if result["statistics"]["state_distribution"]:
                    state_df = pd.DataFrame(result["statistics"]["state_distribution"].items(), columns=["State", "Count"])
                    fig_state = px.bar(state_df, x="State", y="Count", title="State Distribution", color="Count", color_continuous_scale="Greens")
                    st.plotly_chart(fig_state, use_container_width=True)
            else:
                st.warning("No recommendations found for this user.")
    else:
        st.error("Please enter a valid numeric Member ID.")