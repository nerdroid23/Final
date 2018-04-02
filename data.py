import csv


def Writer(data, path):
    with open(path, "a") as c_file:
        write = csv.writer(c_file, delimiter=',')
        write.writerow(data)
