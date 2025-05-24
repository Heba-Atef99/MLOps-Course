import subprocess
from pathlib import Path

if __name__ == '__main__':
    base_path = Path(__file__).parent
    testing_file = base_path / "locustfile.py"

    # # Development
    subprocess.run(["locust", "-f", str(testing_file), "--web-port", "80", "-H", "http://100.24.54.121"])
