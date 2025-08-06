## Requirements：
pip install networkx <br>
pip install scipy <br>
pip install camel-ai==0.2.3 <br> 
pip install mplfinance <br>
pip install squarify <br>
pip install plotly <br>

## Data Generation Pipeline:
### For Chart Image Generation:
1. For Single-plot generation, you should configure OpenAI API key in the 'single_plot_generation_pipeline.py' and 'single_plot_overlay_generation_pipeline.py' and run:
```
python single_plot_generation_pipeline.py
python single_plot_overlay_generation_pipeline.py
```
The generated single plot data and images will be saved in 'ecd_single_plot_charts' folder.

2. For Single-plot diversification, you should configure OpenAI API key in the 'single_plot_diversification.py' and run:
```
python single_plot_diversification.py
```
The diversified images will also be saved under the 'ecd_single_plot_charts' folder.

3. For Combined subplot generation and diversification, you should configure OpenAI API key in the 'combined_subplot_generation_pipeline.py' and run:
```
python combined_subplot_generation_pipeline.py
```
The generated combined subplot data and diversified images will be saved in 'ecd_combined_subplot_charts'.

4. For Figure Size Post-processing, you should configure OpenAI API key and folder_path of your own code_path and run:
```
python figure_size_post_processing.py
```

### For Chart Image Rating:
For chart image rating, you should also configure the OpenAI API key and run:
```
python chart_image_filtering.py --scoring_type 'visual_clarity'
python chart_image_filtering.py --scoring_type 'semantic_coherence'
```
After the chart image rating step, you can simply filter images that ratings are over the average rating of 'visual_clarity' and 'semantic_coherence'.

### For Chart QA Generation:
For Chart QA generation, you should also configure the OpenAI API key / code folder_path and run the following py files to generate the 'descriptive' and 'reasoning'-related QA pairs:
```
python descriptive_qa_generation.py
python reasoning_qa_generation.py
```
After the QA generation step, you can also simply filter QAs that gpt's rating > 5.

Upon completion of the aforementioned steps, a custom ECD dataset can be successfully constructed.










