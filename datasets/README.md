## ECD dataset
You need to download the ECD dataset from the link: 'https://huggingface.co/datasets/ChartFoundation/ECD-10k-Images' and put the "images" and "ECD_QAs_All.json" under this folder.

After the download the data, you should configure the image path in the 'convert_to_format.py' and run:
```
python convert_to_format.py
```
You will get two files: 'ECD_qa_data_all_formatted_for_llamafactory.json' and 'ECD_qa_data_all_formatted.json' for the subsequent MLLMs training.