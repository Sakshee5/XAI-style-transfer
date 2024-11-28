# Art Meets AI: An XAI-Driven Visual Exploration of Neural Style Transfer

## Project Overview

### What is Neural Style Transfer?
Neural Style Transfer (NST) is a computer vision technique that allows you to reimagine images by blending the content of one image with the artistic style of another. Imagine taking a photograph and rendering it in the style of a Van Gogh painting or a Picasso cubist composition!

### Key Features of This Application
- **Content Reconstruction**: Understand how neural networks perceive and reconstruct image content
- **Style Reconstruction**: Visualize how networks capture artistic styles through Gram matrices
- **Interactive Style Transfer**: Blend content and style images with real-time parameter tuning
- **Comprehensive Insights**: Explore the technical nuances of the NST process

### Core XAI Components
1. **Streamlit Interface**: `nst_app.py`
    - Provides an interactive, user-friendly web application with intuitive parameter controls that allow non-experts to explore neural network behavior.
    - Decomposition of Style and Content Reconstruction for better understanding of how the model separates content from style.
    - Visualizes the entire content reconstruction, style reconstruction and style transfer processes seperately.
    - Provides context and explanations for technical parameters

2. **Transparency through Visualization**
    - Layer-wise Feature Extraction: Shows hierarchical feature learning
    - Feature Map Exploration: Reveals what different neural network layers "see"
    - Gram Matrix Visualization: Explains style representation
    - Iterative Reconstruction: Shows how neural networks progressively understand and reconstruct images
    - Grad-CAM Visualizations: Highlights which image regions most influence style and content transfer
    - Comparative Visualization: Side-by-side comparisons of original and reconstructed images

3. **Interaction-Driven Interpretability**
    - Interactive Parameters: Users can modify parameters and observe real-time effects
    - Optimization Process Transparency: Tracks losses during the optimization process gives insight into how the image is evolving to meet objective of minimizing relevant loss function.
    - Progress Bars: To track the reconstruction/style transfer process after training.


## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup Steps
```bash
# Clone the repository
git clone https://github.com/Sakshee5/XAI-style-transfer.git
cd XAI-style-transfer

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run nst_app.py
```

## How to Use

- Navigate through tabs: Home, Content Reconstruction, Style Reconstruction, Neural Style Transfer, Insights
- Upload content and style images (as required)
- Adjust parameters like model, optimizer, iterations and tab specific parameters
- Explore the visual transformations step-by-step

## References

- https://github.com/pytorch/examples/tree/main/fast_neural_style
- https://github.com/gordicaleksa/pytorch-neural-style-transfer
- https://www.youtube.com/watch?v=S78LQebx6jo&list=PLBoQnSflObcmbfshq9oNs41vODgXG-608
