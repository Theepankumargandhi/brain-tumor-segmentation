This project uses a ResNeXt50-UNet model to segment brain tumors from MRI images. The model is deployed as a web app built with Streamlit, allowing users to upload MRI images and instantly view the tumor segmentation mask.

Key Features
Model: ResNeXt50 encoder with a UNet-style decoder for pixel-level tumor segmentation.

Interactive Web App: Built with Streamlit to easily upload MRI images and get segmentation results.

Training Dataset: LGG MRI dataset from Kaggle used to train the model for tumor detection.


git clone https://github.com/<your-username>/brain-tumor-segmentation.git
Install dependencies:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run app.py
Upload an MRI image to see the predicted tumor mask.
The model uses a combination of ResNeXt50 for feature extraction and a UNet decoder for segmentation.
The output is a binary mask indicating tumor presence.

