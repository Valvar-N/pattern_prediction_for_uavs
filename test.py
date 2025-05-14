import utils
import pandas as pd

file = utils.find_file_path("structured_datas", "processed_67.xlsx")
df = pd.read_excel(file)
print(df.info())