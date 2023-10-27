import csv
import json
import pandas as pd
import argparse

def gather_args():
    parser = argparse.ArgumentParser(description='Takes trial data csv from psiturk and converts it to readable rows.')
    parser.add_argument('--input', '-i', default='trialdata.csv', action='store', type=str, help='Location of the file to read data from.')
    parser.add_argument('--output', '-o', default='cleantrialdata.csv', action='store', type=str, help='Location to write clean data to.')
    
    args = parser.parse_args()
    return args

def main():
    args = gather_args()
    trialdata = []
    with open(args.input) as fp:
        reader = csv.reader(fp)
        for row in reader:
            uniqueid = row[0]
            data = json.loads(row[3])
            data['uniqueid'] = uniqueid
            trialdata.append(data)    
    pd.DataFrame(trialdata).to_csv(args.output)

if __name__ == '__main__':
    main()