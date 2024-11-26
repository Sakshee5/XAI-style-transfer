import streamlit as st
from reconstruct_image_from_representation import reconstruct_image_from_representation
from neural_style_transfer import neural_style_transfer
from PIL import Image

def set_config(content_img=None, 
               style_img=None, 
               model='vgg19', 
               optimizer='lbfgs', 
               feature_map_index=2, 
               content_weight=1e5, 
               style_weight=3e4, 
               tv_weight=1e0, 
               init_method='content', 
               noise="white"
               ):

    config = {
        "content_img": content_img,
        "style_img": style_img,
        "model": model,
        "optimizer": optimizer,
        "content_feature_map_index": feature_map_index,
        "content_weight":content_weight,
        "style_weight": style_weight,
        "tv_weight": tv_weight,
        "init_method": init_method,
        "noise": noise,
    }
    return config


st.set_page_config(layout="wide", page_icon="test_images/mosaic.jpg", page_title='NST')
st.title("Art Meets AI: Demystifying Neural Style Transfer")

with st.sidebar:
    st.header("Navigation")
    tab = st.radio("Choose a tab:", ["Home", "Content Reconstruction", "Style Reconstruction", "Neural Style Transfer", "Insights"])

    st.header("User Inputs")

    model = st.radio("Select a model:", ["vgg19", "vgg16"], 
                     help="VGG-19 has more layers and may capture more detailed features, but it trains slower. VGG-16 is lighter and faster but might be less precise.")
    
    optimizer = st.radio("Select an optimizer:", ['lbfgs', 'adam'],
                         help="LBFGS is a more precise optimizer but requires more memory. Adam is efficent but takes longer to converge.")

if tab == "Home":
    with open("description.md", "r") as f:
        markdown_text = f.read()

    st.write(markdown_text)


if tab == "Content Reconstruction":
    st.subheader("Content Reconstruction")
    st.markdown("""
Visualize how the reconstruction process starts with noise and iteratively refines the image to resemble the original content by reducing content loss. Upload an image, select a feature map index to visualize and click "Start Reconstruction" to observe how the content is recreated progressively.""")

    content_image = st.file_uploader("Upload Content Image (.png, .jpeg, .jpg)", type=["png", "jpeg", "jpg"], key='content_tab')

    if content_image:
        content_image = Image.open(content_image)
        st.image(content_image, width=250)
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            feature_map_index = st.slider("Feature Map Index", min_value=1, max_value=5, value=1)

        with col2:
            st.markdown("""Choose which layer's feature representations to visualize. **Lower indices (e.g., 1, 2):** Display features from early layers, such as edges or textures. **Higher indices (e.g., 4, 5):** Show features from deeper layers, like shapes or overall structure. This does not change how the reconstruction is performed but highlights the selected layer's output.""")

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            init_noise = st.selectbox("Choose initial noise type", ['white', 'gaussian'])

        with col4:
            st.markdown("""Choose the random noise initialization to start reconstruction""")
        st.markdown("---")


        start_content_reconstruction = st.button('Start Reconstruction', key='content_reconstruction')

    if content_image and start_content_reconstruction:
        st.write("Reconstructing content from noise! Watch the progress below...")
        config = set_config(content_img=content_image, style_img=None, feature_map_index=feature_map_index, noise=init_noise)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Feature Maps")
            feature_map_placeholder = st.empty() 
        
        with col2:
            st.write("Content Reconstruction Progress")
            video_placeholder = st.empty()

        with col3:
            st.write("Original Content Image")
            st.image(content_image, use_container_width=True)

        reconstruct_image_from_representation(config, feature_map_placeholder, video_placeholder)

elif tab == "Style Reconstruction":
    st.subheader("Style Reconstruction")
    st.markdown("""
This process insight into how neural networks learn and represent stylistic aspects of an image, separate from its content.Style features are captured using **Gram Matrices**, which represent the textures, patterns, and color distributions of an image. Upload an image and click "Start Reconstruction" to observe how the style is recreated progressively.""")
    
    style_image = st.file_uploader("Upload Style Image (.png, .jpeg, .jpg)", type=["png", "jpeg", "jpg"], key='style_tab')

    if style_image:
        style_image = Image.open(style_image)
        st.image(style_image, width=250)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            init_noise = st.selectbox("Choose initial noise type", ['white', 'gaussian'])

        with col2:
            st.markdown("""Choose the random noise initialization to start reconstruction""")
        st.markdown("---")

        start_style_reconstruction = st.button('Start Reconstruction', key='style_reconstruction')

    if style_image and start_style_reconstruction:
        st.write("Reconstructing style from noise! Watch the progress below...")
        config = set_config(content_img=None, style_img=style_image, noise=init_noise)
        
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("Gram Matrices")
            gram_matrices_placeholder = st.empty() 

        with col2:
            st.write("Style Reconstruction Progress")
            style_video_placeholder = st.empty()

        with col3:
            st.write("Original Image")
            st.image(style_image, use_container_width=True)

        reconstruct_image_from_representation(config, gram_matrices_placeholder, style_video_placeholder)

elif tab == "Neural Style Transfer":
    st.subheader("Neural Style Transfer")
    st.markdown("""Upload a content image and a style image. Then hypertune parameters and click "Start Style Transfer" to observe how the style is transferred progressively. Please refer to the insights tab for in-depth comparison between how various parameters affect the result.""")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Content Image")
        content_image = st.file_uploader("Upload Content Image (.png, .jpeg, .jpg)", type=["png", "jpeg", "jpg"], key='content_tab2')

        if content_image:
            content_image = Image.open(content_image)
            st.image(content_image, width=250)

    with col2:
        st.subheader("Style Image")
        style_image = st.file_uploader("Upload Style Image (.png, .jpeg, .jpg)", type=["png", "jpeg", "jpg"], key='style_tab2')

        if style_image:
            style_image = Image.open(style_image)
            st.image(style_image, width=250)

    if content_image and style_image:

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            content_weight = st.slider("Content Weight", min_value=1e3, max_value=1e6, value=1e5, step=1e3, format="%e")

        with col2:
            st.markdown("""Determines how strongly the structural details of the content image are preserved during reconstruction. Use a high value to closely resemble the content image's structure.  
            """)

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            style_weight = st.slider("Style Weight", min_value=1e3, max_value=1e5, value=3e4, step=1e3, format="%e")

        with col4:
            st.markdown("""Controls how strongly the style patterns of the style image influence the reconstruction.Increase this to prioritize reproducing style features (e.g., brushstrokes, textures).  
            """)

        st.markdown("---")
        col5, col6 = st.columns(2)

        with col5:
            tv_weight = st.slider("Total Variation Weight", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

        with col6:
            st.markdown("""Encourages smoothness in the reconstructed image by reducing noise. Higher values result in smoother images but may blur fine details.  
            """)

        st.markdown("---")
        col7, col8 = st.columns(2)

        with col7:
            init_method = st.selectbox("Initialization Method", ['random', 'content', 'style'], index=1)

        with col8:
            st.markdown("""Determines the starting point for reconstruction:  
- **Random**: Starts with noise, gaussian or white.  
- **Content**: Starts with the content image for structural fidelity.  
- **Style**: Starts with the style image for stylistic dominance.""")
        st.markdown('---')

        if init_method == "random":
            col9, col10 = st.columns(2)

            with col9:
                init_noise = st.selectbox("Choose initial noise type", ['white', 'gaussian'])

            with col10:
                st.markdown("""Choose the random noise initialization to start reconstruction""")
            st.markdown("---")
        else:
            init_noise = None


        start_style_transfer = st.button('Start Style Transfer', key='style_transfer')

        if start_style_transfer and content_image and style_image:
            st.write("Transferring style! Watch the progress below...")
            config = set_config(content_img=content_image, style_img=style_image, content_weight=content_weight, style_weight=style_weight, tv_weight=tv_weight, init_method=init_method, noise=init_noise)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.write("Content Image")
                st.image(content_image, use_container_width=True)

            with col2:
                st.write("Neural Style Transfer Progress")
                style_transfer_video_placeholder = st.empty()

            with col3:
                st.write("Style Image")
                st.image(style_image, use_container_width=True)

            neural_style_transfer(config, style_transfer_video_placeholder)


if tab=="Insights":
    with open("insights.md", "r") as f:
        markdown_text = f.read()

    st.markdown(markdown_text)