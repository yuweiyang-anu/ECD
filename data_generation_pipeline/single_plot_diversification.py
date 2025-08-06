import os
import glob
import random
import inspect
import matplotlib
import threading
import matplotlib.pyplot as plt
from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import ChatGPTConfig
from single_plot_generation_pipeline import *
from single_plot_overlay_generation_pipeline import *
from concurrent.futures import ThreadPoolExecutor, as_completed

matplotlib.use('Agg')

####################################################
# You need modify this part of your own API information
os.environ["OPENAI_BASE_URL"] = "base_url_of_openai"
os.environ["API_VERSION"] = "api_version"
os.environ["OPENAI_API_KEY"] = "your_openai_key"
os.environ["DEPLOYMENT_NAME"] = "gpt-4o"
####################################################

save_lock = threading.Lock()

chart_type_list = [
    'line', 'bar', 'pie', 'area', 'error_point', 'treemap', 'graph', 'density',
    'histogram', 'box', 'bubble', 'candlestick', 'heatmap', 'radar', 'rose',
    'quiver', '3d', 'error_bar', 'scatter', 'violin', 'contour', 'funnel',
    'scatter_and_histogram', 'scatter_and_density', 'hex_and_histogram',
    'histogram_and_density', 'bar_and_line', 'box_and_violin', 'pie_and_bar'
]

agent_diversification = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=1.0).as_dict(),
)

agent_debug = ModelFactory.create(
    model_platform=ModelPlatformType.OPENAI,
    model_type=ModelType.GPT_4O,
    model_config_dict=ChatGPTConfig(temperature=0).as_dict(),
)

# Assistant 1: use for single-plot code diverstification
assistant_for_diversification = BaseMessage.make_assistant_message(
    role_name="Assistant",
    content=f"""
### Intelligent Assistant Prompt: Single Plot Code Synthesis and Diversification Helper

You are a 'Single Plot Code Synthesis and Diversification Assistant.' Your task is to synthesize an appropriate single plot code based on the predefined chart function and calling example provided by the user, ensuring that it follows the user's diversification strategy to make the synthesized code sufficiently varied.

**User-Provided Plot Function:**
```python
{{single_plot_function}}
```

**User-Provided Calling Example:**
```python
{{calling_example}}
```

**Synthesized Code Example:**
```python
{{code_body}}
```
""",
)

# Assistant 2: use for code debug
assistant_for_debug = BaseMessage.make_assistant_message(
    role_name="Assistant",
    content=f"""
### Intelligent Assistant Prompt: Code Debug Helper

You are a 'Code Debug Assistant.' Your task is to identify and fix issues in the user's code based on any provided errors, ensuring it works correctly.

**User-Provided Code:**
```python
{{user_code}}
```

**Error Message:**
{{error_message}}

**Returned Code:**
```python
{{whole_code_after_fix}}
```
""",
)

theme_selection = [
    "Economics", "Psychology", "Sociology", "Biology", "Education",
    "Engineering", "Law", "Astronomy", "Computer_Science", "Geography",
    "Physics", "Chemistry", "History", "Environmental_Science", "Anthropology",
    "Media_and_Journalism", "Mathematics", "Statistics", "Finance", "Medicine",
    "Art_and_Design", "Agriculture", "Linguistics", "Architecture", "Sports"
]


def verify_code(code):
    exec(code, globals())


def process_plot(select_chart_type, max_tries=1):

    synthesis_div_assistant = ChatAgent(assistant_for_diversification,
                                        model=agent_diversification,
                                        token_limit=32768)

    synthesis_combined_debug_assistant = ChatAgent(assistant_for_debug,
                                                   model=agent_debug,
                                                   token_limit=32768)

    if '_and_' in select_chart_type:
        single_type = 'plot_combination_' + select_chart_type + '_plots_overlay'
    else:
        single_type = 'plot_' + select_chart_type + '_chart'

    single_plot_function = inspect.getsource(globals()[single_type])

    txt_dir = f"./ecd_single_plot_charts/{single_type}/txt/"
    div_png_dir = f"./ecd_single_plot_charts/{single_type}/div_png/"
    div_code_dir = f"./ecd_single_plot_charts/{single_type}/div_code/"
    os.makedirs(div_code_dir, exist_ok=True)
    os.makedirs(div_png_dir, exist_ok=True)

    print('====Diversification Stage====: Start diversifying the code')

    for filename in os.listdir(txt_dir):
        if filename.endswith('.txt'):
            file_path = os.path.join(txt_dir, filename)
            with open(file_path, 'r') as file:
                calling_example = file.read()

            ds_index, diversification_strategy = random.choice(
                list(
                    enumerate([
                        "Add text labels, annotations, arrows, uncertainty bars, threshold lines (which should have specific names, not just 'threshold'; these could include curves like exponential/logarithmic curves, and not always be straight lines), or highlights (e.g., using a circle or highlighting a range in a specific color) to emphasize key data points, trends, or regions, ensuring that these annotations are contextually relevant and not generic. The number of items added can exceed one.",
                        "Modify the font styles, colors or sizes significantly for titles, labels or ticks.",
                        "Use gradient fills, area shading (typically along the line itself, within a defined range above and below it, must not between the line and the x-axis), or transparency effects to enhance depth. Additionally, fine-tune grid lines, background colors, and shadow effects to improve visual appeal.",
                        "(If Applicable) Remove axis borders for a cleaner, modern look.",
                        "(If Applicable) Incorporate a **zoomed-in inset** of a particular section, ensuring the area is appropriately sized (usually a very small size) and placed, making sure that the elements are visually separated without overlapping."
                    ])))
            print('===========Diversification Strategy================:',
                  diversification_strategy)

            save_name = filename[:-4]
            if any(save_name in file for file in os.listdir(div_png_dir)):
                continue
            else:
                for file in glob.glob(div_code_dir + "*.py"):
                    if save_name in file:
                        os.remove(file)

            div_png_save_path = f"{div_png_dir}{save_name}_ds{ds_index}.png"
            div_code_save_path = f"{div_code_dir}{save_name}_ds{ds_index}.py"

            user_prompt_combined = f"""
                {single_plot_function}
                **Provided calling example for single plot**: {calling_example}. The data/labels/legends in the chart must be preserved at the diversification stage.\n
                """

            user_prompt_combined += f"""
                **Diversification Strategy for Single Plot Chart Code:** {diversification_strategy}.
                **save_path:** {div_png_save_path}.
                **Important Considerations:**
                    --Code Diversity: Provide multiple styles and techniques for modifying the chart, using different libraries such as Seaborn to enhance the code and visual presentation. Must avoid returning results as **functions like 'def' or function calls**, and instead, provide complete executable code.
                    --Layout and Organization: You must follow the **save_path and diversification strategy** and ensure clear separation (without any overlap), alignment and labeling for readability. Additionally, feel free to adjust the figsize to ensure that all elements are fully visible and can be displayed.
                    --Original Data Preservation: You are required to create the code, without modifying or ignoring the original chart data. It is worth noting that for the line_num and bar_num charts, the number annotations should not be modified or removed. 
                **Note:** Be sure not to use any interactive elements that cannot be saved. If chart type is Funnel, use plotly lib for implementation.
                """

            for _ in range(max_tries):
                user_msg = BaseMessage.make_user_message(
                    role_name="User",
                    content=user_prompt_combined,
                )

                # Get the response containing the generated docstring
                response = synthesis_div_assistant.step(user_msg)

                print('===Response by the combined assistant===:', response)
                # Extract the generated python code from the response
                generated_diverse_code = response.msg.content.split(
                    "```python\n")[1].split("\n```")[0]
                print('===Diversified Code:===', generated_diverse_code)

                try:
                    verify_code(generated_diverse_code)
                    with open(div_code_save_path, 'w') as com_code_file:
                        com_code_file.write(generated_diverse_code)
                    plt.close()
                    break
                except Exception as e:
                    # print('Error:', e)
                    print("===Start Debugging===")

                    user_prompt_debug = f"""
                        **Error Code:** {generated_diverse_code},
                        **Error Message:** {e}.
                        You should return all the complete, working code, not just the modified part.
                    """
                    user_msg_debug = BaseMessage.make_user_message(
                        role_name="User",
                        content=user_prompt_debug,
                    )
                    response_refined = synthesis_combined_debug_assistant.step(
                        user_msg_debug)

                    print('repsonse_refined:', response_refined)
                    # Extract the generated python code from the response
                    refined_combined_code = response_refined.msg.content.split(
                        "```python\n")[1].split("\n```")[0]
                    print('code_refined:', refined_combined_code)

                    try:
                        verify_code(refined_combined_code)
                        with open(div_code_save_path,
                                  'w') as com_code_file_ref:
                            com_code_file_ref.write(refined_combined_code)
                        plt.close()
                    except Exception as e1:
                        print('Debug Error:', e1)
                        continue


def run_concurrent_tasks(plot_types, max_workers=1):

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for plot_type in plot_types:
            # Submit the task to the executor
            futures.append(executor.submit(process_plot, plot_type))

        # Optional: process results as they complete
        for future in as_completed(futures):
            try:
                future.result(
                )  # Can raise exceptions if there was an error in the thread
            except Exception as e:
                print(f"Error in thread: {e}")


if __name__ == "__main__":
    # Configure plot types and iterations: total combinations: 300 * 12 = 3600
    selected_types = chart_type_list
    max_concurrent_workers = 1  # Configurable number of concurrent workers

    # # Start concurrent tasks
    run_concurrent_tasks(selected_types, max_workers=max_concurrent_workers)
