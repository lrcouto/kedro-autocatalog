from openai import OpenAI


client = OpenAI()


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

    ### Instructions:
    - If the file looks like a real data file (text, structured, binary, etc), return the most appropriate Kedro dataset class (e.g. `text.TextDataset`) on a single line.
    - If the file clearly isn't data (e.g. .gitkeep, README, etc), return `SKIP`.

    Do not include explanations. Return only the class name or `SKIP`.

    ---
    Filename: {filename}

    Sample content:
    {file_sample}
    ---
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    result = response.choices[0].message.content.strip()

    print(f"[LLM Inference for {filename}] → {repr(result)}")

    normalized = result.lower().replace("`", "").strip()

    if "skip" in normalized or "none" in normalized:
        return None

    if "." in normalized and "dataset" in normalized:
        return result.strip("` ")

    return None
