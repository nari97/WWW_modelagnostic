import os

if __name__ == "__main__":
    files = os.listdir("C:\\Users\\nk1581\\Downloads\\Logs\\")

    filename = "C:\\Users\\nk1581\\Downloads\\Logs\\" + files[0]
    f1 = open(filename, "r")
    result = open("D:\PhD\Work\EmbeddingInterpretibility\Interpretibility\Results\MinedRules\FB15K\\TuckER_materialized.tsv", "w+")
    num_lines = sum(1 for line in open(filename, "r"))
    print("My brotha")
    for ctr in range(0, num_lines-3):
            line = f1.readline()
            result.write(line)
    f1.close()
    for index in range(1, len(files)):
        filename = "C:\\Users\\nk1581\\Downloads\\Logs\\" + files[index]
        num_lines = sum(1 for line in open(filename, "r"))
        f = open(filename, "r")
        for line in f:
            if line[0].isdigit() and "rules" not in line:
                result.write(line)

        f.close()

    result.write("??\n??\n??\n")
    result.close()

