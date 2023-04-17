# References:
# https://medium.com/@bofenghuang7/what-i-learned-from-whisper-fine-tuning-event-2a68dab1862
# https://github.com/huggingface/community-events
# https://huggingface.co/blog/fine-tune-whisper
from khmernltk import word_tokenize
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments,Seq2SeqTrainer
from datasets import Audio, DatasetDict, concatenate_datasets, load_dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
import evaluate

OUTPUT_DIR = "./outputs/whisper-tiny-khmer"
MODEL_NAME = "Whisper Tiny Khmer - Seanghay Yath"
MODEL_ID = "openai/whisper-tiny"
MODEL_LANGUAGE = "khmer"

AUDIO_COLUMN_NAME = "audio"
TEXT_COLUMN_NAME = "sentence"

metric = evaluate.load("wer")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
processor = WhisperProcessor.from_pretrained(MODEL_ID, language=MODEL_LANGUAGE, task="transcribe")
tokenizer = processor.tokenizer

preprocessing_num_workers = 30

max_input_length = 30
min_input_length = 0

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False

def transform_khmer_sentence(ds):
    transcription = word_tokenize(ds['transcription'], return_tokens=False, separator=" ")
    return {"transcription": transcription }

def normalize_dataset(ds, audio_column_name=None, text_column_name=None):
    if audio_column_name is not None and audio_column_name != AUDIO_COLUMN_NAME:
        ds = ds.rename_column(audio_column_name, AUDIO_COLUMN_NAME)
    if text_column_name is not None and text_column_name != TEXT_COLUMN_NAME:
        ds = ds.rename_column(text_column_name, TEXT_COLUMN_NAME)
    # resample to the same sampling rate
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    # normalise columns to ["audio", "sentence"]
    ds = ds.remove_columns(set(ds.features.keys()) - set([AUDIO_COLUMN_NAME, TEXT_COLUMN_NAME]))
    return ds

google_fleurs_train_ds = load_dataset("google/fleurs", "km_kh", split="train+validation", use_auth_token=True)
google_fleurs_train_ds = google_fleurs_train_ds.map(transform_khmer_sentence)

google_fleurs_test_ds = load_dataset("google/fleurs", "km_kh", split="test", use_auth_token=True)
google_fleurs_test_ds = google_fleurs_test_ds.map(transform_khmer_sentence)

openslr_train_ds = load_dataset("openslr", "SLR42", split="train", use_auth_token=True)

kmcs_train_ds = load_dataset("seanghay/kmcs", split="train", use_auth_token=True)
kmcs_train_ds = kmcs_train_ds.map(transform_khmer_sentence)

aug_train_ds = load_dataset("seanghay/km-augmented-16-combined", split="train", use_auth_token=True)

raw_datasets = DatasetDict()
raw_datasets['train'] = concatenate_datasets([
  normalize_dataset(google_fleurs_train_ds, audio_column_name="audio", text_column_name="transcription"),
  normalize_dataset(openslr_train_ds, audio_column_name="audio", text_column_name="sentence"),
  normalize_dataset(kmcs_train_ds, audio_column_name="audio", text_column_name="transcription"),
  
  aug_train_ds # augmented dataset
])

raw_datasets['train'] = raw_datasets['train'].shuffle(seed=10)
raw_datasets['eval'] = normalize_dataset(google_fleurs_test_ds, audio_column_name="audio", text_column_name="transcription")

print(raw_datasets)

def prepare_dataset(batch):
    # load
    audio = batch[AUDIO_COLUMN_NAME]
    # compute log-Mel input features from input audio array
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]

    # process targets
    input_str = batch[TEXT_COLUMN_NAME]
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(input_str).input_ids

    return batch

vectorized_datasets = raw_datasets.map(
    prepare_dataset,
    num_proc=preprocessing_num_workers,
    remove_columns=next(iter(raw_datasets.values())).column_names,
    desc="preprocess dataset",
)


def is_audio_in_length_range(length):
    return length > min_input_length and length < max_input_length


vectorized_datasets = vectorized_datasets.filter(
    is_audio_in_length_range, num_proc=preprocessing_num_workers, input_columns=["input_length"]
)

max_label_length = model.config.max_length
def is_labels_in_length_range(labels):
    return len(labels) < max_label_length

vectorized_datasets = vectorized_datasets.filter(
    is_labels_in_length_range, num_proc=preprocessing_num_workers, input_columns=["labels"]
)

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        # convert to tensors
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad label ids to the max length in the batch
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    warmup_steps=800,
    max_steps=8000,
    learning_rate=6.25e-6,
    weight_decay=0.01,
    gradient_checkpointing=True,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    report_to=["tensorboard"],
    evaluation_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=vectorized_datasets["train"],
    eval_dataset=vectorized_datasets["eval"],
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

model.save_pretrained(training_args.output_dir)
processor.save_pretrained(training_args.output_dir)
trainer.push_to_hub()
