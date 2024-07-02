from load_model_for_inference import model, tokenizer
from rag_module import final_result
result = final_result("what is substance abuse suppport?", model, tokenizer)
print("Response:", result)