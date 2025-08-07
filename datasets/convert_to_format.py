import json
import random

# Load the original JSON file
with open("ECD_QAs_All.json", "r") as f:
    raw_data = json.load(f)

# Shuffle with fixed seed
random.seed(42)
random.shuffle(raw_data)

# Convert format
converted = []
for idx, item in enumerate(raw_data):
    qa_obj = {
        "id":
        str(idx),
        "image":
        f"{item['image_id']}.png",
        "conversations": [{
            "from": "human",
            "value": f"<image>\n{item['question']}"
        }, {
            "from": "gpt",
            "value": item['answer']
        }]
    }
    converted.append(qa_obj)

# Save to output JSON
with open("ECD_qa_data_all_formatted.json", "w") as fout:
    json.dump(converted, fout, indent=2)

print(
    f"Converted {len(converted)} QA entries to 'ECD_qa_data_all_formatted.json'"
)

# Convert format 2
converted_for_llama_factory = []
for idx, item in enumerate(raw_data):
    qa_obj_llama_factory = {
        "conversations": [{
            "from": "human",
            "value": f"<image>{item['question']}"
        }, {
            "from": "gpt",
            "value": item['answer']
        }],
        "images":
        "ECD/datasets/images/" +
        f"{item['image_id']}.png",  # you need modify the image path here
    }
    converted_for_llama_factory.append(qa_obj_llama_factory)

# Save to output JSON
with open("ECD_qa_data_all_formatted_for_llamafactory.json", "w") as fout:
    json.dump(converted_for_llama_factory, fout, indent=2)

print(
    f"Converted {len(converted_for_llama_factory)} QA entries to 'ECD_qa_data_all_formatted_for_llamafactory.json'"
)
