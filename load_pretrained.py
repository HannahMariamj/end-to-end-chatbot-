from transformers import AutoModelForCausalLM, AutoTokenizer

# Define the pre-trained model name and task type
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
task_type = 'text-generation'

# Load the pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token