import warnings
import streamlit as st

warnings.filterwarnings("ignore")
from src.pipeline.prediction_pipeline import PredictionPipeline

# Set the title of the app
st.title("Insurance Premium Prediction App")

# Create a text input widget
col1, col2, col3 = st.columns(3)
with col1:
    # age = st.text_input('Enter the age:')
    age = st.slider("Pick an age", 18, 70)
    if age != None:
        st.write("Selected age:", age)
with col2:
    # children_number = st.text_input('Enter the number of children:')
    children_number = st.slider("Select the number of children", 0, 7)
    st.write("Number of children:", children_number)

with col3:
    bmi = st.text_input("Enter the bmi:")

col11, col21, col31 = st.columns(3)
with col11:
    gender = st.selectbox("Gender:", ["Select Gender", "Male", "Female"])
with col21:
    smoker = st.selectbox("Smoker:", ["Are you smoker", "No", "Yes"])
with col31:
    region = st.selectbox(
        "Region:", ["Select Region", "southwest", "southeast", "northwest", "northeast"]
    )


col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.write("")

with col2:
    st.write("")

with col3:
    st.write("")


def colored_text(text, color):
    return f"<span style='color:{color};'>{text}</span>"


def unselected_items(gender_flag, smoker_flag, region_flag):
    message = []
    if gender_flag == False:
        message.append("Gender")
    if smoker_flag == False:
        message.append("Smoker")
    if region_flag == False:
        message.append("Region")

    if len(message) == 1:
        return message[0]

    else:
        return ", ".join(message[:-1]) + " and " + message[-1]


def isnumeric(num):
    try:
        float(num)
        return False
    except Exception as e:
        return True


if st.button("Submit Text"):

    bmi_flag = True
    gender_flag = True
    smoker_flag = True
    region_flag = True

    if bmi == "":
        bmi_flag = False

    if gender == "Select Gender":
        gender_flag = False

    if smoker == "Are you smoker":
        smoker_flag = False

    if region == "Select Region":
        region_flag = False

    if (
        (bmi_flag == False)
        or (gender_flag == False)
        or (smoker_flag == False)
        or (region_flag == False)
    ):
        if (bmi_flag == False) and (
            (gender_flag == False) or (smoker_flag == False) or (region_flag == False)
        ):
            message = unselected_items(gender_flag, smoker_flag, region_flag)
            st.write(f"Please enter the BMI and select {message}.")

        elif bmi_flag == False:
            st.write("Please enter the BMI.")

        elif (gender_flag == False) or (smoker_flag == False) or (region_flag == False):
            message = unselected_items(gender_flag, smoker_flag, region_flag)
            st.write(f"Please select {message}.")

    elif isnumeric(bmi):
        st.write("BMI must be a number (int or float).")

    else:
        st.write("You entered:")

        inputs = [age, children_number, bmi, gender, smoker, region]
        input_names = ["Age", "Children Number", "BMI", "Gender", "Smoker", "Region"]

        num_pairs_per_line = 2

        # Calculate the number of required columns
        num_columns = num_pairs_per_line * 2

        # Adjust the column proportions
        columns = st.columns([1, 1] * num_pairs_per_line)

        for i in range(len(inputs)):
            col_name = columns[2 * i % num_columns]
            col_input = columns[2 * i % num_columns + 1]

            with col_name:
                st.markdown(
                    colored_text(input_names[i], "white"), unsafe_allow_html=True
                )
            with col_input:
                st.markdown(colored_text(inputs[i], "green"), unsafe_allow_html=True)

        single_data_point = {
            "region": [region],
            "sex": [gender.lower()],
            "smoker": [smoker.lower()],
            "age": [int(age)],
            "bmi": [float(bmi)],
        }

        data_point = [single_data_point, int(children_number)]
        pred_pipeline = PredictionPipeline()
        predicted_val = pred_pipeline.prediction(data_point)

        st.write("PREDICTED AMOUNT: ", round(predicted_val, 2))
