import csv
import matplotlib.pyplot as plt


name_to_index = {
    "swebench": 0,
    "classeval": 1,
    "avatar": 2,
    "cruxeval": 3,
    "humaneval": 4,
    "real": 5
}

def donuts_two_rings(results, label, save_path):
    plt.cla()
    def make_labels(data):
        return [v if int(v)>28 else "" for v in data]
    # Data for each ring
    data1 = results["strong"]   # Inner ring
    data2 = results["weak"]   # Middle ring

    # Colors (optional, for visual clarity)
    colors =  ['salmon', 'orange', 'gold', 'lightgreen', "skyblue", "lightgrey"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('equal')  # Equal aspect ratio

    # Common width for each ring
    width = 0.4

    # First (inner) ring
    ax.pie(data1, radius=1-width, colors=colors,
        wedgeprops=dict(width=width, edgecolor='white'),
        labels=make_labels(data1), labeldistance=0.44, textprops={'fontsize': 32})

    # Second (middle) ring
    if sum(data2) > 0:
        ax.pie(data2, radius=1, colors=colors,
            wedgeprops=dict(width=width, edgecolor='white'),
            labels=make_labels(data2), labeldistance=0.70, textprops={'fontsize': 32})

    ax.set_title(label, fontsize=16)
    plt.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0)
    
def get_benchmark(s):
    if "cruxeval" in s:
        return "cruxeval"
    elif "HumanEval" in s:
        return "humaneval"
    elif "rebuttal" in s:
        return "real"
    elif "Avatar" in s:
        return "avatar"
    elif "swebench" in s:
        return "swebench"
    else:
        return "classeval"
def load_data(index):
    strong_list = [0,0,0,0,0,0]
    weak_list = [0,0,0,0,0,0]
    csv_path = "./analysis/tagged.csv"
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            data = list(line)
            benchmark = get_benchmark(data[0])
            m = data[index]
            if m == "strong":
                strong_list[name_to_index[benchmark]] += 1
            elif m == "weak":
                weak_list[name_to_index[benchmark]] += 1
    result = {
        'dataset': ['swebench', 'classeval', 'avatar', 'cruxeval', 'humaneval', 'real'],
        'strong': strong_list,
        'weak': weak_list
    }
    savepath = f"./analysis/figs/difficulty/M{index}.png"
    print(result)
    donuts_two_rings(result, f"M{index} Difficulty Breakdown", savepath)
    
if __name__ == "__main__":
    load_data(9)