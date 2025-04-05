import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open('spam123.pkl', 'rb'))
cv = pickle.load(open('vec123.pkl', 'rb'))

# Main function
def main():
    # Page title and description
    st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="wide")
    st.title("📧 Email Spam Classification App")
    st.markdown("""
        Welcome to the Email Spam Classifier! This application uses **Machine Learning** to classify emails as `Spam` or `Not Spam`.
        Simply paste the email content below to get started.  
        """)
    st.sidebar.title("About the App")
    st.sidebar.info("""
        - This app uses a pre-trained machine learning model.
        - Enter an email to check if it's spam or not.
        - Built with ❤️ using **Streamlit**.
    """)
    
    # Input section
    st.subheader("🔍 Classify Your Email")
    user_input = st.text_area(
        "Enter the email content below:",
        placeholder="Paste your email here...",
        height=200
    )

    # Classify button
    if st.button("🔗 Classify Email"):
        if user_input.strip():  # Ensure input is not empty
            with st.spinner("Analyzing email..."):
                data = [user_input]
                vec = cv.transform(data).toarray()
                result = model.predict(vec)
                
                # Display results
                if result[0] == 0:
                    st.success("✅ This is NOT a Spam Email!")
                    st.balloons()
                else:
                    st.error("🚫 This is a SPAM Email!")
        else:
            st.warning("⚠️ Please enter email content to classify!")

    # Footer
    st.markdown("---")
    st.markdown("""
        Developed by **Aditya Utsav**  
        For queries, contact: [adityautsav123456@gmail.com](mailto:adityautsav123456@gmail.com)  
        
    """)

# Run the app
if __name__ == "__main__":
    main()

