import numpy as np

with open("covid19_day_wise.csv", "r", encoding="utf-8") as f:
    data = f.readline()
    # data = f.readlines()
print(data)
covid = {
    "date": [],
    "data": [],
    "header": [h for h in data[0].strip().split(",")[1:]]
}
print(covid)

for row in data[1:]:
    spilt_row = row.strip().split(",")
    covid["date"].append(spilt_row[0])
    covid["data"].append([float(n) for n in spilt_row[1:]])