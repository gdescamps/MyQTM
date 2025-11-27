from dotenv import load_dotenv

from src.data_pipeline import run_pipeline
from src.utils.printlog import PrintLog

if __name__ == "__main__":

    # Load environment variables from .env file
    # This ensures sensitive information like API keys is securely loaded into the environment.
    load_dotenv()

    local_log = PrintLog(extra_name="_data", log_time=True, enable=False)
    run_pipeline(log_local=local_log)
