import streamlit as st
from predict_category import get_predicted_class
from BERT_classifier import get_model
from labels import Labels
from predict_answer import generate_post
from pylatexenc.latex2text import LatexNodes2Text

def convert_latex_to_text(latex_str):
    return LatexNodes2Text().latex_to_text(latex_str)


# Main app layout
def main():
    # st.title("Math Helper App")
    # st.subheader("Classify your math question and get help to solve it")
    st.markdown("<h1 style='color:#1f77b4;'>🧮 Math Helper App</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color:gray;'>Classify and solve math problems instantly!</h3>", unsafe_allow_html=True)

    # Input: math question
    user_question = st.text_area("Enter your math problem:", height=150)
    if st.button("Analyze"):
        if not user_question.strip():
            st.warning("Please enter a question.")
       
        tokenizer,model=get_model()
        # Predict class and confidence
        predicted_class_id, conf = get_predicted_class(user_question,tokenizer,model)
        # Use the dictionary
        label_instance = Labels()
        label_map = label_instance.class_names
        print(f"Predicted label: {label_map[predicted_class_id]} (Confidence: {conf:.2%})")
        st.success(f"Predicted Category: **{label_map[predicted_class_id]}** with confidence: **{conf:.2f}**")
        response=generate_post(user_question)
        print(response)
        st.markdown("### Step-by-Step Solution")
        st.markdown(f"```\n{response}\n```")


# Run the app
if __name__ == "__main__":
    main()

        
