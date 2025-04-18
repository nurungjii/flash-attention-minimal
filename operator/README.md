Run `flash-attn-minimal-llms.py` to test LLMs with our substitute flash attn code. To do so, we need the below file structure:
```
.
├── minimal_attn/
│   └── csrc/
│       ├── default.cpp
│       └── cuda/
│           └── flash.cu
│   ├── __init__.py
│   ├── ops.py
├── setup.py
```

### Setup
Note that we need python3.10 so that pybind works. Currently have python3.12. So, in terminal:
```bash
source /project/engineering/anaconda3/etc/profile.d/conda.sh
conda create -n py310 python=3.10
conda activate py310
conda install jupyter ipykernel
pip3 install torch torchvision torchaudio
pip install transformers
pip install Ninja   # for building cuda
pip install --no-build-isolation -e .  # for our new operator
python -m ipykernel install --user --name py310 --display-name "Python 3.10 (py310)"  # to use as a kernel in future jupyter notebooks.
```

### Testing Imlementation
Run `flash-attn-minimal-llms.py` to ensure our implementation works within Python. If the sanity checks pass, we are good to go.

### Inference
Now, we will integrate our attention mechanism into (an) LLM(s) for inference. To do so, we will modify the modeling files in HuggingFace. We start with GPT-2 as it is simple and can run in low memory.
First, 
```bash
mkdir hf_home
export HF_HOME=hf_home/
```
so we don't take up too much storage in our home directory, which may be limited.

To implement our `minimal_attn` implementation into GPT-2, we edit the `modeling_gpt2.py` file from HuggingFace.

*NOTE: To debug this, I am using a python terminal as jupyter was weird with my env. But if I make a change in `modeling_gpt2.py` (e.g., to debug), it doesn't just register; we need to restart the terminal for it to take effect.*

Now, using `run_llms.ipynb` to load and test the model with our implementations.