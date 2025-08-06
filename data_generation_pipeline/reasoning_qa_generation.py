import re
import os
import io
import base64
from PIL import Image
import json
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ProcessPoolExecutor, as_completed


def extract_and_complete_json(text):
    pattern = re.compile(r'\{(?:[^{}]*"image_id": "[^"]+")[^{}]*\}', re.DOTALL)
    matches = pattern.findall(text)

    json_objects = []
    for match in matches:
        try:
            json_obj = json.loads(match)
            json_objects.append(json_obj)
        except json.JSONDecodeError:
            continue
    return json_objects


def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def get_response(image, prompt):
    client = OpenAI(
        api_key="your_api_key"
    )

    png_file = Image.open(image)
    base64_image = encode_image(png_file)

    iteration = 0
    while iteration < 5:
        try:
            response = client.chat.completions.create(
                messages=[{
                    "role":
                    "user",
                    "content": [{
                        "type": "text",
                        "text": prompt
                    }, {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }]
                }],
                model="gpt-4o",
                n=1,
                max_tokens=4096,
                temperature=0.7,
                top_p=1,
                seed=None,
            ).choices[0].message.content
            break
        except Exception as e:
            iteration += 1
            print(f"===== Retry: {iteration} =====")
            print(f"Error occurs when calling API: {e}")
            continue
    return response


def find_python_files(directory):
    python_file_paths = []

    # Traverse the specified directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):  # Check the file extension
                full_path = os.path.join(root, file)  # Get the full path
                python_file_paths.append(full_path)  # Add to the list

    return python_file_paths


def process_single_qa_task(task_info):
    i, original_code_path = task_info
    QA_name = os.path.basename(original_code_path)[:-3]

    corres_img_path = original_code_path.replace("code", "png").replace(
        ".py", ".png")
    # img_path = corres_img_path.replace("\\", "/")
    output_qa_path = os.path.join(
        os.path.split(original_code_path)[0].split("code")[0], "QA_Reason")
    os.makedirs(output_qa_path, exist_ok=True)

    file_path = os.path.join(output_qa_path, f'{QA_name}_reason.json')
    if not os.path.exists(file_path):
        pass
    else:
        print(
            f'=== Fig {i+1}: Generating QA for {QA_name}, Img Path: {corres_img_path} ==='
        )

        try:
            with open(original_code_path, 'r') as f1:
                ori_data_code = f1.read()

            prompt = rf"""As an expert in chart mathematical analysis, you are provided with the original chart code as follows:
                            
                ```python
                {ori_data_code}
                ```

                You are tasked with examining a given chart code and generating a series of in-depth **Analytical Reasoning-based** question-answer pairs. Additionally, you have been given the chart image. The chart image will help you verify if the questions can be fully answered from it.

                ### Generation Requirements:
                1. Generate a diverse set of questions based on the chart data. The questions should cover a variety of aspects, such as comparisons, ratios, rankings, trends, range, overlaps, extreme values, relationships, highlights, annotations, shading area or calculations. You can freely combine any of above aspects to create a challenging question.

                2. Ensure that the phrasing of the questions is varied and complex. Avoid using the same simple question structure or repetitive wording. Use a variety of question types, including but not limited to 'Is/Are', 'What', 'How', 'How many (times)', 'Which', 'Where', 'when' and 'If'. Incorporate different sentence structures, such as direct, indirect, conditional (e.g., 'If/Assuming ...', 'Excluding/Discarding ...', 'As ....'), rhetorical, **selective** (e.g., '... or ...?') and **negated** questions (e.g., 'Which is not ...?'). 

                3. Include nested questions that require layered reasoning, as well as declarative sentences that prompt identification or clarification (e.g., 'Compared to/Comparing/Considering/Focusing on ...', 'In terms of/Given ...', 'Referring to ...', 'In/On/At/Across/Around/For/Of/Over/From/Between/By/Among/With/During/Towards ...', 'Based on ...', etc.). You can use a combination of one declarative instruction and one question, but must not to combine two questions.

                4. Be sure to ask questions that address different dimensions of the data, such as time, categories, labels, names, values, and relative changes, rather than focusing only on the largest or smallest values (consider including **middle-ranked** values/labels such as 'the second', 'the third' etc., for a more balanced analysis). 

                5. If the chart contains subplots, prioritize questions that **involve reasoning across multiple subplots**, or focus on a specific subplot (by indicating its location, name, or ID) to explore detailed aspects of the data.

                6. The generated questions should **exhibit clear visual references**, with an emphasis on the nuanced variations in details. **Be sure to ask how the legend labels or colorbar interact with the data.** The corresponding answers can be text or numbers, but not always number of values.

                ### Response Format:
                Please use the following JSON format to ensure clarity:

                ```json
                {{
                    {{
                        "image_id": "{QA_name}_1",
                        "question": "Provide your question here",
                        "rationale": "Provide your concise analysis and calculation procedures (MUST enumerate all relevant elements with concrete values and show **detailed calculations with specific values**)",
                        "answer": "Provide complete final answer here",
                        "rating": "Rate your confidence in this question, rationale and answer [0-5], Integer",
                    }},
                    {{
                        "image_id": "{QA_name}_2",
                        ... 
                    }},
                    ...
                }}
                ```

                ## Notes:
                1. Focus only on aspects that can be inferred visually from the chart image. Do not refer to facts or terms that directly suggest the content is based on the code rather than the chart image itself. 
                2. Frame questions to encourage mathematical reasoning and logical deductions, avoiding sole reliance on chart data. 
                3. **MUST ENSURE** the rationale provides **numerical/text recognition or calculations**, converting visual questions into recognition- and calculation-based questions to support clear and logical reasoning. 
                4. The total number of questions should be **10 to 15**, and feel free to rephrase your questions to enhance diversity.
                5. **MUST INCLUDE** one instance of "Not Applicable" for answer (different from "No") when appropriate. If a question relates to specific content that is missing in the image (especially for legend and colorbar, such as when asking about legend (e.g., no legend), values of colorbar (e.g., no colorbar)), answer "Not Applicable". Questions categorized as 'Not Applicable' should begin with 'What' and be generated based on your judgement.
                6. Do not include questions about **alpha value, transparency, opacity, padding, spacing, grid, width, viewing angle, role, rotation, function (e.g.,'np.random'), font size, figsize, interpretation, visualization, significance, impact, save filename and save_path** in the code. Focus only on the meaningful visual elements in the chart and avoid mentioning 'code' in the questions, rationales and answers.
                7. Ensure the image_id numbers increment sequentially, such as _1, _2, _3, _4, and so on. Keep the ```json``` formatting intact.
                """

            response = get_response(corres_img_path, prompt)

            json_response = response.split("```json")[1].split(
                "```")[0].strip()
            json_objects = extract_and_complete_json(json_response)

            with open(os.path.join(output_qa_path,
                                    f'{QA_name}_reason.json'),
                        'w',
                        encoding='utf-8') as f:
                json.dump(json_objects, f, indent=4)

        except Exception as e:
            print(f'Error in QA generation for {QA_name}: {e}')


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    directory_to_search = 'your_path_of_code_files'
    python_files = find_python_files(directory_to_search)
    task_list = [(i, file) for i, file in enumerate(python_files)]

    print(f"Found {len(task_list)} Python files for QA generation.")

    max_workers = min(1, os.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_qa_task, task) for task in task_list
        ]
        for _ in tqdm(as_completed(futures),
                      total=len(futures),
                      desc="Processing QA Tasks"):
            pass
