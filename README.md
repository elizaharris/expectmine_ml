### (Optional) Use conda environment

```bash
conda create --name mlinvitrotox
```

```bash
conda activate mlinvitrotox
```

```bash
conda install pip
```

### Install dependencies

```bash 
pip install -r requirements.txt
```

### Running pipeline:
- Replace `python` with `python3` or `py` depending on your python version
Main ML pipeline:
```bash 
python ml\src\pipeline\ml_pipeline.py
```
Metadata ML pipeline:
```bash
python ml\src\pipeline\metadata_ml_pipeline.py
```
Post ML pipeline:
```bash
python ml\src\pipeline\post_ml_pipeline.py
```
Feature importance:
```bash
python ml\src\pipeline\feature_importance.py
```

### Running streamlit app:
```bash
streamlit run ml\src\app\Results.py
```



