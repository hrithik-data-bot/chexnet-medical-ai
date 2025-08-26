"""base folder settings"""

import sys
# import dotenv
import pandas as pd

# access to src/python
sys.path.insert(0, "../src/python")

# Load all environment variables from settings.env
# dotenv.load_dotenv('../../settings.env')

# DataFrame display options
pd.options.display.max_colwidth = 2000
pd.options.display.max_rows = 1000
