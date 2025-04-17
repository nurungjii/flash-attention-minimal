Run `flash-attn-minimal-llms.ipynb` to test LLMs with our substitute flash attn code. To do so, we need the below file structure:
```
.
├── minimal_attn/
│   └── csrc/
│       └── cuda/
│           └── flash.cu  # holds the operator defining the flash attention operation
├── setup.py
```