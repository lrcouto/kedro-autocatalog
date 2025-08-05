from openai import OpenAI

client = OpenAI()


SYSTEM_PROMPT = """
You are an expert in the Kedro framework.

Always skip files that are clearly not datasets, such as (but not limited to):
- .gitkeep
- README files
- timestamped folders (e.g. 2025-08-05T04.17.40.084Z)
- Kedro versioned dataset subfolders

These files are not actual data sources and should not be added to the catalog.

Only respond with:
- A single valid Kedro dataset class (like `pandas.CSVDataset`)
- Or `SKIP` if the file should be ignored
"""


def infer_kedro_dataset_type(filename: str, file_sample: str) -> str:
    prompt = f"""
    You are an expert in the Kedro framework.

    You will receive a file name and a sample of its contents. Your task is to infer the most appropriate Kedro dataset type for loading this file.

    Kedro supports many dataset types, including:
        "api.APIDataset",
        "biosequence.BioSequenceDataset",
        "dask.CSVDataset",
        "dask.ParquetDataset",
        "databricks.ManagedTableDataset",
        "email.EmailMessageDataset",
        "geopandas.GenericDataset",
        "holoviews.HoloviewsWriter",
        "huggingface.HFDataset",
        "huggingface.HFTransformerPipelineDataset",
        "ibis.FileDataset",
        "ibis.TableDataset",
        "json.JSONDataset",
        "matlab.MatlabDataset",
        "matplotlib.MatplotlibDataset",
        "matplotlib.MatplotlibWriter",
        "networkx.GMLDataset",
        "networkx.GraphMLDataset",
        "networkx.JSONDataset",
        "openxlml.DocxDataset",
        "pandas.CSVDataset",
        "pandas.DeltaTableDataset",
        "pandas.ExcelDataset",
        "pandas.FeatherDataset",
        "pandas.GBQTableDataset",
        "pandas.HDFDataset",
        "pandas.JSONDataset",
        "pandas.ParquetDataset",
        "pandas.SQLTableDataset",
        "pandas.SQLQueryDataset",
        "pandas.XMLDataset",
        "partitions.IncrementalDataset",
        "partitions.PartitionedDataset",
        "pillow.ImageDataset",
        "pickle.PickleDataset",
        "plotly.HTMLDataset",
        "plotly.JSONDataset",
        "plotly.PlotlyDataset",
        "polars.CSVDataset",
        "polars.EagerPolarsDataset",
        "polars.LazyPolarsDataset",
        "redis.PickleDataset",
        "snowflake.SnowparkTableDataset",
        "spark.SparkDataset",
        "spark.SparkHiveDataset",
        "spark.SparkJDBCDataset",
        "spark.SparkStreamingDataset",
        "svmlight.SVMLightDataset",
        "tensorflow.TensorFlowModelDataset",
        "text.TextDataset",
        "yaml.YAMLDataset"

    Return only the Kedro dataset class or `SKIP`.

    Filename: {filename}
    Sample content:
    {file_sample}
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    result = response.choices[0].message.content.strip()

    print(f"[LLM Inference for {filename}] → {repr(result)}")

    normalized = result.lower().replace("`", "").strip()

    if "skip" in normalized or "none" in normalized:
        return None

    if "." in normalized and "dataset" in normalized:
        return result.strip("` ")

    return None
