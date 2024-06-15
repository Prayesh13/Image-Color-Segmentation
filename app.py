import streamlit as st
import matplotlib.image
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
import numpy as np


# Function to convert RGB to Hex
def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


# Function to create an individual color swatch
def create_color_swatch(color, swatch_size=(300, 150)):
    swatch = Image.new("RGB", swatch_size, tuple(color.astype(int)))
    return swatch


# Title of the app
st.title("Image Uploader and Dominant Colors Finder with KMeans")

# File uploader allows only image files
file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if file is not None:
    # Slider to select number of clusters
    n_clusters = st.slider('Select number of colors', 1, 10, 3)  # Default is 3 clusters

    btn = st.button("Submit")
    if btn:
        # Read the uploaded image file
        image = matplotlib.image.imread(file)

        # Display the original image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Reshape the image array
        X = image.reshape(-1, 3)

        # Apply KMeans clustering to find specified number of colors
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++',random_state=0)
        kmeans.fit(X)
        dominant_colors = kmeans.cluster_centers_

        # Display each dominant color as an image with its hex code
        st.write("Dominant Colors (Image and Hex):")
        for i, color in enumerate(dominant_colors):
            rgb = tuple(color.astype(int))
            hex_color = rgb_to_hex(rgb)
            swatch = create_color_swatch(color)

            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(swatch, use_column_width=True)
            with col2:
                st.write("\n\n")
                st.subheader(f" :  {hex_color}")

    st.write("----------------------------------")
    st.subheader("About")
    st.write("The Color Image Segmentation Project is a web-based application "
             "designed to identify and display dominant colors in an image using KMeans clustering. "
             "This tool allows users to upload any image and dynamically segment it into its most prominent colors. The resulting palette of colors can be used for various applications "
             "such as color analysis, design inspiration, and artistic exploration.")

else:
    st.warning("Please upload an image file.")
