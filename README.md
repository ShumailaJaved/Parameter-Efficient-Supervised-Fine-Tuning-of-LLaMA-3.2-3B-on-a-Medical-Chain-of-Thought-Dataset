# 🧠 Fine-Tuning LLaMA-3.2 for Medical Conversational AI (with Unsloth)

This project fine-tunes the `unsloth/Llama-3.2-3B-Instruct` model for a specialized **medical question-answering task** using the **Unsloth** framework. The dataset used is `blue-blues/medical_cot`, which includes medically grounded prompts with system/user/assistant roles in structured templates.

The training script utilizes LoRA adapters (parameter-efficient fine-tuning), and evaluates performance with the **ROUGE-L metric** before and after training.

---

## 🚀 Key Features

- ✅ Loads and quantizes **LLaMA 3.2 3B** or 1B via `unsloth`
- ✅ Parameter-efficient fine-tuning via **LoRA (Low-Rank Adaptation)**
- ✅ Uses **Hugging Face `SFTTrainer`** for supervised fine-tuning
- ✅ Trains on a **medical CoT dataset** with conversational role structure
- ✅ Includes **ROUGE-L** evaluation before and after training
- ✅ Uploads fine-tuned model to Hugging Face Hub

---

## 🧩 Model & Dataset

- **Base Model**: [`unsloth/Llama-3.2-3B-Instruct`](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct)
- **Training Dataset**: [`blue-blues/medical_cot`](https://huggingface.co/datasets/blue-blues/medical_cot)
- **Output**: LoRA-finetuned model with medical reasoning capability

---

## 📦 Installation

Run the following to install dependencies:

```bash
pip install unsloth
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
pip install transformers==4.51.3
pip install rouge-score
