import pandas as pd
import seaborn as sns
import matplotlib as plt

def main():

    # Load the dataset
    dataDir = "./glass+identification/glass.data"
    column_names = ["ID", "RI", "NA2O", "MGO", "AL2O3", "SIO2", "K2O", "CAO", "BAO", "FE2O3", "TYPE"]
    glassDataset = pd.read_csv(dataDir, index_col=0, names=column_names )

    # Display dataset information
    print(glassDataset.info())
    print(glassDataset.head())

    # Prepare the features and labels
    X = glassDataset.drop(columns=["TYPE"])
    y = glassDataset["TYPE"]

    # Show correlation between each set of features and the labels
    #corr_matrix = X.corr()
    #sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    #print(X.corrwith(y))
    #plt.pyplot.savefig("dataset_feature_correlation")
    
    import svm
    import naiveBayes
    import knn
    import decisionTrees
     

if __name__ == "__main__":
    main() 
    
