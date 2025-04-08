#  Brain Tumor Segmentation using ResNeXt50-UNet (Streamlit App)

This project is focused on segmenting brain tumors from MRI images using a deep learning model that combines **ResNeXt50** as the encoder and a **UNet-style** decoder for pixel-level tumor prediction. The model is trained on the **LGG MRI Segmentation Dataset** and predicts tumor regions effectively.

A **Streamlit web app** allows users to upload MRI images, process them through the trained model, and view the segmented brain tumor mask in real-time.

##  Features
- **Upload MRI images** and see the predicted brain tumor segmentation mask.
- Uses **ResNeXt50** as the encoder and **UNet** as the decoder for tumor segmentation.
- Built with **Streamlit** for a simple and interactive user interface.
- Visualizes the **segmented tumor mask** over the original MRI image.
- Efficient preprocessing and model inference.

## 📁 Folder Structure

brain-tumor-segmentation/
│
├── README.md                     # Project documentation
├── requirements.txt              # Dependency list
├── app/
│   └── streamlit_app.py          # Streamlit user interface
├── model.py                      # Model definition (ResNeXt50-UNet)
├── outputs/                      # Diagrams, results, or processed images
├── model_brain_mri.pth           # Trained model file 

git clone https://github.com/YourUsername/brain-tumor-segmentation.git
cd brain-tumor-segmentation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py

Use Cases:
Medical imaging: Segmenting brain tumors for diagnostic purposes.

Healthcare AI: Automating the process of tumor detection in MRI images.

Deep learning in healthcare: A demonstration of applying modern deep learning techniques to medical data.
