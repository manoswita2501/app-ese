import streamlit as st
from PIL import Image

# Page layout
st.title('Image Processing')

# Upload image
uploaded_image = st.file_uploader("Choose an image of your choice...", type=["jpg", "jpeg", "png"])

# Display uploaded image
if uploaded_image is not None:
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Image processing options
    selected_option = st.selectbox('Select an image processing technique:',
                                   ['Resize', 'Grayscale Conversion', 'Image Cropping', 'Image Rotation'])

    # Image processing functions
    if selected_option == 'Resize':
        new_size = st.slider('Select new size:', 10, 1000, 300)
        image = Image.open(uploaded_image)
        image_resized = image.resize((new_size, new_size))
        st.image(image_resized, caption='Resized Image', use_column_width=True)

    elif selected_option == 'Grayscale Conversion':
        image = Image.open(uploaded_image)
        grayscale_image = image.convert('L')
        st.image(grayscale_image, caption='Grayscale Image', use_column_width=True)

    elif selected_option == 'Image Cropping':
        left = st.slider('Left:', 0, 500, 0)
        top = st.slider('Top:', 0, 500, 0)
        right = st.slider('Right:', 0, 500, 500)
        bottom = st.slider('Bottom:', 0, 500, 500)
        image = Image.open(uploaded_image)
        cropped_image = image.crop((left, top, right, bottom))
        st.image(cropped_image, caption='Cropped Image', use_column_width=True)

    elif selected_option == 'Image Rotation':
        angle = st.slider('Select rotation angle (degrees):', -180, 180, 0)
        image = Image.open(uploaded_image)
        rotated_image = image.rotate(angle)
        st.image(rotated_image, caption='Rotated Image', use_column_width=True)
