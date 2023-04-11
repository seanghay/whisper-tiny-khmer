
## Install deps

```shell
conda create -n whisper transformers
conda activate whisper
pip install -r requirements.txt
```


## Start training

Login with `huggingface.co`

```shell
huggingface-cli login
```



Start

```shell
python3 seq2seq.py
```

> Adjust these if got CUDA error: out of memory!

```python
per_device_train_batch_size=64, # decrease by 2x
per_device_eval_batch_size=32, # decrease by 2x
gradient_accumulation_steps=1, # increase 2x

# effective_batch_size = gradient_accumulation_steps x per_device_train_batch_size
```
