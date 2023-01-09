import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import json


def dist_citations(folder_path,start_year,end_year):

    stat_table = df['Total'].describe()
    stat_table.to_excel(f"{folder_path}_stat_table_{start_year}_{end_year}.xlsx")
    print(stat_table)
    tot_citations = np.array(df['Total'].to_list())

    plt.hist(tot_citations,bins = 100,histtype='step',density=True)
    plt.xlabel('Number of citations')
    plt.yscale("log")
    plt.xscale("log")
    plt.xlim(right=3000)
    plt.title("Distribution of the number of citations for hep-th articles (2010-15)")
    plt.savefig(f"{folder_path}citation_distribution_{start_year}_{end_year}.png")


def excel(folder_path, start_year,end_year):
    df.to_excel(f"{folder_path}total_citation_year_th_{start_year}_{end_year}.xlsx")
    print("Excel file created")

def report_json():

    reports = glob.glob("reports/*.json")
    mean = int(df['Total'].mean())
    classifier = []
    precision = {}
    recall = {}
    f1_score = {}
    f1_micro = {}

    for i, report in enumerate(reports):
        with open(report, "r") as file:
            data = json.load(file)
            classifier.append(data['Estimator'])
            precision[i] = [100*data['A']['precision'],100*data['B']['precision'],100*data['C']['precision']]
            recall[i] = [100*data['A']['recall'],100*data['B']['recall'],100*data['C']['recall']]
            f1_score[i] = [100*data['A']['f1-score'],100*data['B']['f1-score'],100*data['C']['f1-score']]
            f1_micro[data['Estimator']] = 100*data['f1_micro']

         
    abs = [f'[0,{mean})',f'[{mean},100]','> 100 citations']

    plt.figure(0)
    df_micro = pd.DataFrame({'Classifiers':list(f1_micro.values())},index=classifier)
    cp = df_micro.plot.bar(rot=0,legend=False)
    plt.title('F1 micro')
    plt.savefig('reports/figures/f1_micro_all_classifiers.png')
    print('reports/figures/f1_micro_all_classifiers.png has been saved')
    
    for i, clf in enumerate(classifier):
        plt.figure(i+1)
        df_prf = pd.DataFrame({'Precision':precision[i],'Recall':recall[i],'F1':f1_score[i]},index=abs)
        ax = df_prf.plot.bar(rot=0)
        plt.title(f'{clf} Performance')
        plt.savefig(f'reports/figures/{clf}_cat_prf.png')
        print(f'reports/figures/{clf}_cat_prf.png has been saved')

        
    

if __name__ == "__main__":
    df = pd.read_pickle("data/arxiv_id_total_citation_year_th_2010_2015.pkl")
    #excel("reports/",2010,2015)
    #dist_citations("reports/figures/",2010,2015)

    report_json()
