# LLM model scaling pipeline

## For (s)LLM model scaling

### Dataset

Cosmecca Recipe dataset

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/recipe-ai-test.git
cd recipe-ai-test

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10 -y
conda activate myenv

# install requirements
pip install -r requirements.txt
```

### .env file setting

```shell
PROJECT_DIR={PROJECT_DIR}
CONNECTED_DIR={CONNECTED_DIR}
DEVICES={DEVICES}

API_KEY={API_KEY}
REMOTE_API_BASE_RECOMMEND={REMOTE_API_BASE_RECOMMEND}
REMOTE_API_BASE_REPORT={REMOTE_API_BASE_REPORT}

SERVER_HOST={SERVER_HOST}
RECOMMEND_SERVER_PORT={RECOMMEND_SERVER_PORT}
REPORT_SERVER_PORT={REPORT_SERVER_PORT}
```

### Vector store

* Set up(command)

```shell
python set_vector_store.py
```

* Set up(script)

```shell
bash scripts/set_vector_store.sh
```

### Run(demo)

* Run demo(command)

```shell
streamlit run app.py remote_mode=False
```

* Run demo(script)

```shell
bash scripts/app.sh
```

### Run

* Recommend API(command)

```shell
python -m src.apis.recommend
```

* Recommend API(script)

```shell
bash scripts/recommend_api.sh
```

* Report API(command)

```shell
python -m src.apis.report
```

* Report API(script)

```shell
bash scripts/report_api.sh
```

* Run UI(command)

```shell
streamlit run app.py remote_mode=True
```

* Run UI(script)

```shell
bash scripts/ui.sh
```
