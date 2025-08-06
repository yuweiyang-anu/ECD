import os
import re
import subprocess
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")

####################################################
# You need modify this part of your own API information
client = OpenAI(
    api_key="your_api_key",
    api_version="api_version"
)
####################################################

# Regular expression to extract code blocks
code_block_re = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Function to create the GPT prompt
def create_prompt(original_code, filename):
    return f"""
    You are an experienced Python developer specializing in data visualization and chart rendering. The following Python code generates and saves charts. Some charts may have unnecessarily large or inappropriate sizes, while others may already be optimal. Modify the code as follows:

    1. If the chart size is inappropriate (e.g., unnecessarily large or excessively small), adjust the figsize to a **reasonable resolution** suitable for typical use cases.
    2. If the chart size is already appropriate, leave it unchanged.
    3. Saves with dpi **no higher than 150** (e.g., `dpi=100`).
    4. Guarantee that axis labels, titles, legends and tick labels **do not overlap**:
        • Use `fig.tight_layout()` or `constrained_layout=True`.  
        • Dynamically scale `figsize` or reduce `plt.rcParams["font.size"]` when needed.  
        • Verify that every text element remains readable and non-overlapping.
    5. Do not alter the chart content, style, or any other functionality of the code.
    6. Headless: set backend "Agg" before pyplot and remove all *.show()* calls.
    7. Ensure the modified code is efficient, readable, and adheres to best practices for saving visualization outputs.

    Here is the original code:
    ```python
    {original_code}
    ```
    Respond with the complete modified code, enclosed in Python code block tags for clarity.
    """


# Function to send the code to GPT for modification
def modify_code_with_gpt(file_content, filename):
    prompt = create_prompt(file_content, filename)
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt
        }],
        model="gpt-4o",
        n=1,
        max_tokens=4096,
        temperature=0,
        top_p=1,
    )
    raw = response.choices[0].message.content
    match = code_block_re.search(raw)
    return match.group(1).strip() if match else raw.strip()


# Function to debug the code using GPT
def debug_code_with_gpt(file_content, filename, error_message):
    prompt = f"""
    The following Python code was modified to save images, but it produces an error when executed. Debug the code to fix the error.

    Error message:
    {error_message}

    Here is the code:
    ```python
    {file_content}
    ```
    Respond with the fully debugged code, enclosed in Python code block tags for clarity.
    """
    response = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": prompt
        }],
        model="gpt-4o",
        n=1,
        max_tokens=4096,
        temperature=0,
        top_p=1,
    )
    return response.choices[0].message.content


# Main function to process `.py` files in the `code` directory
def process_files():
    input_dir = "path_of_your_code"  # need configure the path of your code
    output_dir = "path_of_the_output_refined_code" # need configure the path of your output refined code
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("rendered_images", exist_ok=True)

    py_files = [f for f in os.listdir(input_dir)]

    print(f"Number of Python files to process: {len(py_files)}")

    # Single-threaded processing
    for filename in py_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        with open(input_path, "r") as f:
            original_code = f.read()
        modified_code = modify_code_with_gpt(original_code,
                                             os.path.splitext(filename)[0])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(modified_code)

        # Try running the modified code
        try:
            subprocess.run(["python", output_path],
                           check=True,
                           capture_output=True,
                           text=True)
            print(f"{filename}: Successfully executed")
        except subprocess.CalledProcessError as e:
            print(f"{filename}: Error encountered, debugging...")
            debugged_code = debug_code_with_gpt(modified_code,
                                                os.path.splitext(filename)[0],
                                                e.stderr or str(e))
            debugged_code = code_block_re.search(debugged_code).group(
                1).strip()
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(debugged_code)
            try:
                subprocess.run(["python", output_path],
                               check=True,
                               capture_output=True,
                               text=True)
                print(f"{filename}: Fixed and executed successfully")
            except subprocess.CalledProcessError as e2:
                print(f"{filename}: Failed after debugging: {e2.stderr or e2}")


# Run the script
if __name__ == "__main__":
    process_files()