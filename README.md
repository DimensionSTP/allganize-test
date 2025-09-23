# LLM model scaling pipeline

## For (s)LLM model scaling

### Dataset

Cosmecca Recipe dataset

### Quick setup

```bash
# clone project
git clone https://github.com/DimensionSTP/allganize-test.git
cd allganize-test

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
```

### Vector store

* Set up(command)

```shell
python set_vector_store.py
```

### Run(stack RAG results)

* Run demo(command)

```shell
python main.py
```
