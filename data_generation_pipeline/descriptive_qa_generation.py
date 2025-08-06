import re
import os
import io
import json
import base64
from PIL import Image
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

    # print('QA_name:', QA_name)
    corres_img_path = original_code_path.replace("code", "png").replace(
        ".py", ".png")
    # img_path = corres_img_path.replace("\\", "/")
    output_qa_path = os.path.join(
        os.path.split(original_code_path)[0].split("code")[0], "QA_Des")
    os.makedirs(output_qa_path, exist_ok=True)

    file_path = os.path.join(output_qa_path, f'{QA_name}_des.json')
    if not os.path.exists(file_path):
        pass
    else:
        print(
            f'=== Fig {i+1}: Generating QA for {QA_name}, Img Path: {corres_img_path} ==='
        )

        try:
            with open(original_code_path, 'r') as f1:
                ori_data_code = f1.read()

            prompt = rf"""You are a careful and precise observer of the chart. You are provided with the original chart code as follows:

                ```python
                {ori_data_code}
                ```

                Your task is to carefully analyze the chart code details and generate **Basic Descriptive-based** question-answer pairs. Additionally, a chart image is provided to assist in verifying whether the questions can be fully addressed based on the visual information it contains.

                ### Areas to Focus On:
                1. **Textual Content**: This includes elements like titles, subtitles, axis label (e.g., x-axis or y-axis), tick marks (e.g., labeled ticks at extremes, like **the leftmost/rightmost/highest/lowest**, etc.), annotations, or legends (e.g., list of legend labels).

                2. **Numerical Content**: This refers to the count of variables (e.g., lines/bars/instances), the number of ticks across all axes, **differences between numerical tick values on the x- and y-axes**, label count in legends, and the maximum/minimum (or **difference between the maximum and minimum**) and intervals in continuous colorbar, etc. 

                3. **Graphical Content**: This includes visual characteristics such as:
                    - Chart type
                    - Point positions
                    - Line intersections and thresholds
                    - Overall orientation (e.g, "veritical" or "horizontal", not the angle of specific elements)
                    - **Subplot layouts** (e.g., answer in format like "1 by 2", "2 by 4", "3 by 3")
                    - Number of subplots
                    - Highlights and zoom-in inset (e.g., emphasizing specific features for clarity)
                    - General trends and patterns (e.g., "increase," "decrease," "stabilize")

                ### Response Format:
                Please use the following JSON format to ensure clarity:

                ```json
                {{
                    {{
                        "image_id": "{QA_name}_1",
                        "rating": "Rate your confidence in this question and answer [0-5], Integer",
                        "question": "Provide your question here",
                        "answer": "Provide your corresponding complete answer here"   
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
                2. The total number of questions should be **10 to 15**, and feel free to rephrase your questions to enhance diversity.
                3. The QA pairs should include questions involving the enumeration and counting of elements such as legends, ticks (e.g., total across all axes, like x_axis + y_axis), variables, etc. Furthermore, simple calculations are necessary for identifying differences.
                4. Do not include questions about **alpha value, transparency, opacity, padding, spacing, grid, width, viewing angle, rotation, function (e.g.,'np.random'), font size, figsize, save filename and save_path** in the code. Focus only on the meaningful visual elements in the chart and avoid mentioning 'code' in the questions and answers. 
                5. Ensure the image_id numbers increment sequentially, such as _1, _2, _3, _4, and so on. Keep the ```json``` formatting intact.
                """

            response = get_response(corres_img_path, prompt)
            print('response:', response)

            json_response = response.split("```json")[1].split(
                "```")[0].strip()
            json_objects = extract_and_complete_json(json_response)

            with open(os.path.join(output_qa_path, f'{QA_name}_des.json'),
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
