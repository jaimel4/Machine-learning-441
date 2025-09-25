from tabulate import tabulate

def print_table(rows, headers):
    print(tabulate(rows, headers=headers, floatfmt=".4f"))
