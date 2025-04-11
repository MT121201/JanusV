from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "deepseek-ai/Janus-Pro-7B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenizer.save_pretrained("/home/a3ilab01/pretrain/janus-pro-7b")
model.save_pretrained("/home/a3ilab01/pretrain/janus-pro-7b")
