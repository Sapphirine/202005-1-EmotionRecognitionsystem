import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
with open('conf_mat_pickle.pkl', 'rb') as f:
    # print(f)
    df_cm = pickle.load(f)
    #plt.figure(2)
    print(df_cm)
    fig, ax = plt.subplots()
    #print(df_cm)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    sns.heatmap(df_cm, annot=True, fmt='d', cmap=plt.cm.Blues)
    ax.set_ylim(len(df_cm) - 0.5, -0.5)
    plt.tight_layout()
    plt.show()