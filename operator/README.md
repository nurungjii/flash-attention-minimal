Run `flash-attn-minimal-llms.py` to test LLMs with our substitute flash attn code. To do so, we need the below file structure:
```
.
├── minimal_attn/
│   └── csrc/
│       ├── default.cpp
│       └── cuda/
│           └── flash.cu
            └── improved_flash.cu  // our version
│   ├── __init__.py
│   ├── ops.py
├── setup.py
```

### Setup
Note that we need python3.10 so that pybind works. Currently have python3.12. So, in terminal in this directory `flash-attention-minimal/operator`:
```bash
source /project/engineering/anaconda3/etc/profile.d/conda.sh
conda create -n py310 python=3.10
conda activate py310
export PIP_CACHE_DIR=""  # if necessary (since I have it set to nonexistent path)
conda install jupyter ipykernel
pip3 install torch torchvision torchaudio
pip install transformers
pip install Ninja   # for building cuda
pip install --no-build-isolation -e .  # for our new operator -- rerun if changes
python -m ipykernel install --user --name py310 --display-name "Python 3.10 (py310)"  # to use as a kernel in future jupyter notebooks.
```

### Add New Implementation
To add a new implementation, add new functions to `flash.cu` (couldn't figure out how to do new file -- was resulting in the module getting a segfault or something). Ensure you give the function a unique name, e.g., `improved_mha_forward`, then update the `TORCH_LIBRARY*` functions accordingly.
Then, go into `ops.py`, add the name to `__all__`, add new definition of function (and create new buffers if `cudaMalloc` would be used), and implement a corresponding function with your chosen name.

After these changes, ensure to call `pip install --no-build-isolation -e .` again. This new implementation will then still lie in the `minimal_attn` package.

#### Important Note
When you are designing `flash.cu`, it's important *NOT* to use `cudaMalloc`. We were testing and would often get illegal access errors and different results for the forward passes, even though from basic testing (outside of the LLM) the implementation seemed correct. We could not figure this out, so we asked ChatGPT and it said using `cudaMalloc` was actually a bad thing as PyTorch is the one managing the memory, so there could be all sorts of overwrites and illegal accesses if we try to have it managed directly in the new kernel. It actually suggested allocating a memory buffer (an empty tensor) in Python first, then passing it in as a ptr to C++. This makes sense, so we will try to implement it.

### Testing Implementation
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

Now, using `run_llms.ipynb` to load and test the model with our implementations. The way the Jupyter Notebook GPU memory requests work, sometimes I think the blocks will get assigned to invalid memory or something so we get `RuntimeError: CUDA error: an illegal memory access was encountered`. Restarting kernel can fix it.