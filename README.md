This notebook implements a **Multi-Agent System (MAS)** using **LangGraph** to automate the classification and clinical analysis of Diabetic Retinopathy (DR) from retinal fundus images.

The system follows a modular, stateful architecture where specialized agents handle everything from GCS data ingestion to generating final clinical summaries using Gemini 3.5 flash.

---

# Diabetic Retinopathy Classification Pipeline (MAS-AI)

## ## Overview
This project leverages **Google Gemini Vision** and **LangGraph** to process large-scale medical imaging datasets. The pipeline is designed to be "hardened," meaning it includes robust error handling for GCS timeouts, corrupted images, and LLM API rate limits.

### ### Key Features
* **Stateful Orchestration:** Uses `LangGraph` to manage the flow between 6 specialized agents.
* **Vision-Language Integration:** Employs `gemini-2.5-flash` for high-speed image classification.
* **Automated Evaluation:** Generates confusion matrices and classification reports (Accuracy, F1, Precision, Recall) automatically.
* **Clinical Summary:** A final "Output Agent" (`gemini-2.5-pro`) interprets the raw metrics into flowing clinical prose.

---

## ## Architecture
The system is composed of the following nodes:

| Agent | Responsibility |
| :--- | :--- |
| **Input Agent** | Discovers metadata and fetches raw image bytes from Google Cloud Storage. |
| **Preprocessing Agent** | Normalizes color modes (RGB), resizes to $224 \times 224$, and converts to Base64. |
| **Classifier Agent** | Invokes Gemini Vision to grade the image into one of four DR categories. |
| **Tabular Agent** | Records predictions into a structured format and increments the loop index. |
| **Analyzer Agent** | Computes Scikit-Learn metrics and generates a JSON-based statistical report. |
| **Output Agent** | Translates technical metrics into a concise, plain-English clinical explanation. |

---

## ## Data Schema
The graph operates on a shared `ImageClass` state schema:

```python
class ImageClass(TypedDict):
    img_id: str
    ground_truth: str
    gemini_classification: str
    all_images: List[Dict[str, Any]]
    current_index: int
    image_b64: str
    preprocessed_b64: str
    results: List[Dict[str, str]]
    df_json: str
    analysis_metrics: Dict[str, Any]
    nlp_explanation: str
```

---

## ## Setup & Usage

### ### 1. Prerequisites
* A Google Cloud Project with a GCS bucket containing the [EyePACs or similar DR dataset](https://colab.research.google.com/drive/1GZuZcvIxhmOPPX1z2s6z2gcmwbg9C6Re).
* API keys for Gemini (Vertex AI or Google AI Studio) stored in Colab Secrets.

### ### 2. Installation
```bash
pip install -q langchain-google-genai langgraph torch torchvision Pillow pandas scikit-learn
```

### ### 3. Running the Pipeline
The pipeline is initialized with an `initial_state` and executed via:
```python
# Compile the graph
dr_graph = builder.compile()

# Invoke the system
final_state = dr_graph.invoke(initial_state)
```

---

## ## Classification Labels
The model classifies images into one of the following exact categories:
* `NO_DR`: Healthy retina.
* `MI_DR`: Mild Diabetic Retinopathy.
* `MO_DR`: Moderate Diabetic Retinopathy.
* `SE_DR`: Severe Diabetic Retinopathy.

## ## Performance Monitoring
The system includes built-in timing guards. If a GCS download or image transformation takes longer than **30s**, a warning is logged to the console to help identify network bottlenecks or compute limitations.


