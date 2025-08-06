import os
import re
import io
import json
import argparse
import base64
from PIL import Image
from openai import OpenAI

############################## GPT-4o Config ############################################
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format=image.format)
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def get_response(image, prompt):
    # Configuration
    client = OpenAI(
        api_key="your_api_key"
    )

    png_file = Image.open(image)
    base64_image = encode_image(png_file)

    iteration = 0
    while iteration < 1:
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
                max_tokens=2048,
                temperature=0,
                top_p=1,
                seed=None,
            ).choices[0].message.content
            break
        except Exception as e:
            iteration += 1
            response = "Rating: 1"
            print(f"===== Retry: {iteration} =====")
            print(f"Error occurs when calling API: {e}")
            continue
    return response


######################################################################################

prompt_for_visual_clarity = f"""
You're an expert evaluating the **visual clarity** of a chart. Assign a rating from 1 to 5, adhering to the detailed descriptions below:

    **1 point**: The chart is essentially unreadable due to overwhelming issues, such as excessive clutter, severe text overlap, missing legends, or large blank areas, making it impossible to interpret the data accurately.
    **2 points**: The chart has significant design flaws that hinder understanding. Issues like noticeable clutter, text overlap, or missing elements complicate the data's presentation and make it difficult to extract meaningful insights.
    **3 points**: While the chart communicates the data to some extent, it suffers from moderate issues. Mild clutter, misleading scales, or slight text overlap may confuse viewers, affecting overall clarity.
    **4 points**: The chart is largely clear and conveys the data effectively, though minor issues like slight text overlap, small misalignments, or minor clutter might cause slight distractions without greatly affecting comprehension.
    **5 points**: The chart is visually excellent, with precise and well-organized data presentation. It uses space effectively, ensuring that the information is easy to interpret and completely free of any errors.
"""

prompt_for_semantic_coherence = f"""
You're an expert evaluating the **semantic coherence** of a chart. Assign a rating from 1 to 5, adhering to the detailed descriptions below:

    **1 point**: The chart is irrelevant to its intended message or theme. The data is disjointed or disconnected, with no clear connection between elements or subplots. There is no recognizable story, and the chart does not support the overall narrative in any meaningful way.
    **2 points**: The chart's content partially aligns with its theme, but there are significant gaps in how the data is presented. The relationships between elements are weak or unclear, and subplots may seem out of place, making it difficult to extract a clear narrative or insight from the chart.
    **3 points**: The chart generally supports its theme or message, but the connections between data points or subplots are inconsistent or ambiguous. The overall story is somewhat discernible, but there are sections where the data feels disjointed or hard to follow.
    **4 points**: The chart aligns well with its theme, presenting a mostly coherent story. There are clear connections between the main plot and subplots, though some minor ambiguities or distractions might slightly affect the clarity of the narrative. The data supports the theme effectively.
    **5 points**: The chart is highly coherent, with a clear, logical progression of ideas. The data is directly relevant to the theme, and the relationships between the main plot and any subplots are seamless. The chart presents a compelling and easy-to-follow narrative, making the insights immediately apparent.
"""


def extract_scoring(response):
    match = re.search(r'Rating:\s*([1-5])', response)
    if match:
        return int(match.group(1))
    else:
        return 0


def scoring(png_dir_list, output_path, scoring_type="visual_clarity"):

    scoring_list = []
    previous_start_point = 0
    for i, png_dir in enumerate(png_dir_list):
        scoring_dict = {}

        try:
            if scoring_type == "visual_clarity" and 'combination' in png_dir and 'subplots' in png_dir:
                match = re.search(r"\((\d+),\s*(\d+)\)", png_dir)
                layout = "(" + match.group(1) + "," + match.group(2) + ")"
                user_prompt = f"""Layout for current chart: {layout}, you should consider whether the chart image matches the provided layout. Give your rating for the provided chart image now."""
            elif scoring_type == "semantic_coherence":
                pattern = r"\\\d+_([A-Za-z_]+?)(?:_\(|_ds)"
                match = re.search(pattern, png_dir)
                theme = match.group(1)
                user_prompt = f"""Theme for current chart: {theme}, you should consider whether the chart image matches the provied theme. Give your rating for the provided chart image now."""
            else:
                user_prompt = f"""Give your rating for the provided chart image now."""

            if scoring_type == "visual_clarity":
                final_prompt = prompt_for_visual_clarity + '\n' + user_prompt + \
                f"""      
                \nProvide your rating in the format without any explanation: \nRating: (integer value within 1 to 5)
                """
            else:
                final_prompt = prompt_for_semantic_coherence + '\n' + user_prompt + \
                f"""      
                \nProvide your rating in the format without any explanation: \nRating: (integer value within 1 to 5)
                """
        except Exception as e:
            print('error:', e, 'skipping...')
            continue

        # print('final_prompt:', final_prompt)

        response = get_response(png_dir, final_prompt)
        print('response:', response)
        extracted_score = extract_scoring(response)

        if scoring_type == "visual_clarity":
            if i == 0 and os.path.exists(
                    output_path + "scoring_for_benchmark_visual_clarity.json"):
                with open(
                        output_path +
                        "scoring_for_benchmark_visual_clarity.json",
                        'r') as previous_scoring_file:
                    previous_data = json.load(previous_scoring_file)
                scoring_list.extend(previous_data)
                scoring_dict[
                    "image_index"] = f"{i + len(previous_data):06d}"
                previous_start_point = i + len(previous_data)
            else:
                scoring_dict[
                    "image_index"] = f"{i + previous_start_point:06d}"

            print('====Generating Current Image Index:====',
                  scoring_dict["image_index"])
            scoring_dict["image_path"] = png_dir
            scoring_dict["visual_clarity_score"] = extracted_score
            scoring_list.append(scoring_dict)
            with open(
                    output_path + "scoring_for_benchmark_visual_clarity.json",
                    'w') as vc_scoring_file:
                json.dump(scoring_list, vc_scoring_file, indent=4)
        elif scoring_type == "semantic_coherence":
            if i == 0 and os.path.exists(
                    output_path +
                    "scoring_for_benchmark_semantic_coherence.json"):
                with open(
                        output_path +
                        "scoring_for_benchmark_semantic_coherence.json",
                        'r') as previous_scoring_file:
                    previous_data = json.load(previous_scoring_file)
                scoring_list.extend(previous_data)
                scoring_dict[
                    "image_index"] = f"{i + len(previous_data):06d}"
                previous_start_point = i + len(previous_data)
            else:
                scoring_dict[
                    "image_index"] = f"{i + previous_start_point:06d}"

            print('====Generating Current Image Index:====',
                  scoring_dict["image_index"])
            scoring_dict["image_path"] = png_dir
            scoring_dict["semantic_coherence_score"] = extracted_score
            scoring_list.append(scoring_dict)
            with open(
                    output_path +
                    "scoring_for_benchmark_semantic_coherence.json",
                    'w') as sc_scoring_file:
                json.dump(scoring_list, sc_scoring_file, indent=4)


if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(
        description="Process directory paths and scoring types.")

    # Add arguments
    parser.add_argument("--directory_path",
                        type=str,
                        default=".\\data_generation_pipeline\\",
                        help="The path to the directory containing images.")
    parser.add_argument(
        "--scoring_type",
        type=str,
        default="visual_clarity",
        help="The type of scoring to perform. Default is 'visual_clarity'.")

    # Parse arguments
    args = parser.parse_args()

    # Access arguments
    base_directory = args.directory_path
    scoring_type = args.scoring_type

    # Print the parsed arguments (for debugging or confirmation)
    print(f"Directory Path: {base_directory}")
    print(f"Scoring Type: {scoring_type}")

    # Example usage
    # Add your code here to use these arguments, e.g., process the directory or scoring logic
    png_dir_list = []  # Example list to store files
    print(
        f"Processing directory: {base_directory} with scoring type: {scoring_type}"
    )

    # Traverse the directory and its subdirectories
    # Subdirectories to search
    dir_single_plot = os.path.join(base_directory,
                                   "ecd_single_plot_charts")
    dir_combined_subplots = os.path.join(base_directory,
                                         "ecd_combined_subplot_charts")

    # Helper function to collect PNG file paths in specified subdirectories
    def collect_png_files(base_dir, target_sub_dir=None, exclude_sub_dir=None):
        png_paths = []  # Store paths to the PNG files

        # Traverse all subdirectories under the base directory
        for root, dirs, files in os.walk(base_dir):
            # Check if the current path contains the target subdirectory (if specified)
            if target_sub_dir and target_sub_dir not in root:
                continue

            # Exclude paths that contain the exclude_sub_dir (if specified)
            if exclude_sub_dir and exclude_sub_dir in root:
                continue

            # Collect PNG file paths
            for file in files:
                if file.lower().endswith(
                        ".png"):  # Case-insensitive check for .png files
                    full_path = os.path.join(root, file)
                    png_paths.append(full_path)

        return png_paths

    # Collect PNG file paths in benchmark_single_plot_charts under div_png/
    png_dir_list_single_plot = collect_png_files(dir_single_plot,
                                                 target_sub_dir="div_png")

    # Collect PNG file paths in benchmark_combined_subplot_charts under png/ excluding single_png/
    png_dir_list_combined_subplots = collect_png_files(
        dir_combined_subplots,
        target_sub_dir="png",
        exclude_sub_dir="single_png")

    # Combine all PNG file paths into a single list
    png_dir_list = (png_dir_list_single_plot + png_dir_list_combined_subplots)

    print('png_dir_list:', len(png_dir_list))

    if scoring_type == "visual_clarity":
        file_path = base_directory + "scoring_for_ecd_visual_clarity.json"
    else:
        file_path = base_directory + "scoring_for_ecd_semantic_coherence.json"

    existing_image_paths = []
    try:
        # check the existing scoring
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for data_item in data:
            existing_image_paths.append(data_item['image_path'])

        print('existing_image_path:', existing_image_paths)
        remaining_image_paths = [
            item for item in png_dir_list if item not in existing_image_paths
        ]
        print('remianing_png_dir_list:', len(remaining_image_paths))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warrning: {e}, Now Starting From All the PNG files...")
        remaining_image_paths = png_dir_list

    scoring(
        remaining_image_paths,
        base_directory,
        scoring_type,
    )


