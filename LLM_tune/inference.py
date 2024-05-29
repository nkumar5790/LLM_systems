from unsloth import FastLanguageModel
max_seq_length = 2048



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

inputs = tokenizer(
[ 
    alpaca_prompt.format(
        "where does the term sweat equity come from?", # instruction
        "", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
print(tokenizer.batch_decode(outputs))

# print('\n Saving the model\n')
# model.save_pretrained_merged("model_l3", tokenizer, save_method = "merged_4bit_forced",)
# model.push_to_hub_merged("trex5790/model_l3", tokenizer, save_method = "merged_4bit_forced", token = "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")

