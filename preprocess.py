import jieba

def load_raw_data(output_path=None):
    data_file = open("./dataset/train.data", encoding="utf-8-sig")
    label_file = open("./dataset/train.solution", encoding="utf-8-sig")
    emoji_file = open("./dataset/emoji.data", encoding="utf-8-sig")

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
    if output_path != None:
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

def seg_words(dataset, output_path=None):
    new_dataset = []
    for instance in dataset:
        words = jieba.cut(instance[0])
        new_instance = [words, instance[1]]
        new_dataset.append(new_instance)

    #write the new dataset into disk file
    if output_path != None:
        output_file = open(output_path, "w", encoding="utf-8")
        for new_instance in new_dataset:
            #output the label
            output_file.write(str(new_instance[1]) + " ")

            #output the words
            for word in new_instance[0]:
                output_file.write(str(word) + " ")
            output_file.write("\n")
        
        output_file.close()

    return new_dataset

if __name__ == "__main__":
    dataset, class_num = load_raw_data()

    seg_words(dataset, "./dataset/segmented_train.dataset")
