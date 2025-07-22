## Diabetic Retinopathy Classification Pipeline

Modular pipeline for diabetic retinopathy classification using:
- Deep feature extraction with ResNet50
- Feature selection using Particle Swarm Optimization (PSO)
- Classification using SVM / Random Forest
- Interpretability with Grad-CAM
- Cluster visualization using t-SNE
- Interactive dashboard via Streamlit

## Project Structure
- `utils/`: Modular pipeline functions
- `main_pipeline.py`: Orchestration script
- `streamlit_app/`: Grad-CAM & classification dashboard

## Quickstart
```bash
pip install -r requirements.txt
python main_pipeline.py
streamlit run streamlit_app/app.py
