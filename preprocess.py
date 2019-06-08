
def load_raw_data(output_path="null"):
    data_file = open("./dataset/train.data", encoding="utf-8")
    label_file = open("./dataset/train.solution", encoding="utf-8")
    emoji_file = open("./dataset/emoji.data", encoding="utf-8")

    train_data = data_file.readlines()
    train_label = label_file.readlines()
    emoji = emoji_file.readlines()
    emoji_map = {}
    dataset = []

    #construct emoji map: string -> id
    for line in emoji:
        content = str(line).split()
        emoji_map[content[1]] = content[0]

    #combine files to construct dataset
    if len(train_data) != len(train_label):
        print("Invalid data, may be stained")
    for lineno in range(len(train_data)):
        emoji_name = str(train_label[lineno]).rstrip('\n').rstrip('}').lstrip('{')
        instance = [train_data[lineno].strip(), int(emoji_map[emoji_name])]
        dataset.append(instance)

    #write the dataset into disk file
    if output_path != "null":
        output_file = open(output_path, "w", encoding="utf-8")
        for instance in dataset:
            output_file.write(str(instance[1]) + " " + instance[0] + "\n")
        
        output_file.close()

    data_file.close()
    label_file.close()
    emoji_file.close()
    return dataset, len(emoji_map.keys())
    
def class_statistic(dataset, class_num):
    counters = {}
    for i in range(class_num):
        counters[i] = 0

    #count instances for every class
    for instance in dataset:
        counters[instance[1]] = counters[instance[1]] + 1

    return counters

if __name__ == "__main__":
    dataset, class_num = load_raw_data()

    class_data = class_statistic(dataset, class_num)
    for key in range(class_num):
        val = class_data[key]
        print("{0}:{1}".format(key, val))