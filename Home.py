import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Python ESE",
    page_icon="👗",
    layout="wide"
)

# Page layout
st.title('Womens Clothing E-Commerce Dashboard')
         
# Add images
st.image('static/image1.jpg', width=250, caption='Image 1')
st.image('static/image4.jpg', width=250, caption='Image 2')

# Add navigation options
st.sidebar.title('Options:')
selection = st.sidebar.radio("Go to", ['Home', 'About'])

# Add About section
if selection == 'About':
    st.write("""
    **Women’s Clothing E-Commerce dataset**
    
    This is a Women’s Clothing E-Commerce dataset revolving around the reviews written by customers. Its
    nine supportive features offer a great environment to parse out the text through its multiple dimensions.
    Because this is real commercial data, it has been anonymized, and references to the company in the
    review text and body have been replaced with “retailer”.
    
    **Content:**
    
    This dataset includes 23486 rows and 10 feature variables. Each row corresponds to a customer review,
    and includes the variables:
    - Age: Positive Integer variable of the reviewers' age.
    - Title: String variable for the title of the review.
    - Review Text: String variable for the review body.
    - Rating: Positive Ordinal Integer variable for the product score granted by the customer from 1
      Worst to 5 Best.
    - Recommended IND: Binary variable stating where the customer recommends the product
      where 1 is recommended, and 0 is not recommended.
    - Positive Feedback Count: Positive Integer documenting the number of other customers who
      found this review positive.
    - Division Name: Categorical name of the product high-level division.
    - Department Name: Categorical name of the product department name.
    - Class Name: Categorical name of the product class name.
    """)
    
# Add page navigation
if selection == 'Home':
    st.write('Welcome to the Home page')