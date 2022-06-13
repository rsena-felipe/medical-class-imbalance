import my_functions
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, roc_curve, auc, f1_score, precision_score, recall_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def calculate_f1score(input_file, prediction_column, true_label_column):
    """
    Compute F1 weighted for training and validation

    Arguments:
    input_file (str) -- Path to .csv that has the predictions with columns and Prediction, True_Label, Split
    prediction_column (str) -- Name of the prediction column
    true_label_column (str) -- Name of the true label column

    Returns:
    f1_train (float) -- f1 weighted of training 
    f1_val (float) -- f1 weighted of validation
    """
    # Read data and splits into training and vallidation
    df = pd.read_csv(input_file)
    df_train = df[ df["split"] == "training" ]
    df_val = df[ df["split"] == "validation" ]
    df_test = df[ df["split"] == "testing" ]

    # Compute F1 Weighted Training
    y_pred = df_train[prediction_column]
    y_true = df_train[true_label_column]
    f1_train = f1_score(y_true, y_pred, average = "weighted")

    # Compute F1 Weighted Validation
    y_pred = df_val[prediction_column]
    y_true = df_val[true_label_column]
    f1_val = f1_score(y_true, y_pred, average = "weighted")

    # Compute F1 Weighted Testing
    y_pred = df_test[prediction_column]
    y_true = df_test[true_label_column]
    f1_test = f1_score(y_true, y_pred, average = "weighted")

    return f1_train, f1_val, f1_test

def calculate_satisfying_metric(input_file, prediction_column, true_label_column):
    """
    Compute auc, balanced accuracy, precision, recall for training and validation

    Arguments:
    input_file (str) -- Path to .csv that has the predictions with columns and Prediction, True_Label, Split
    """
    # Read data and splits into training and vallidation
    df = pd.read_csv(input_file)
    df_train = df[ df["split"] == "training" ]
    df_val = df[ df["split"] == "validation" ]
    df_test = df[ df["split"] == "testing" ]

    # Training
    # Divide y_pred and y_true 
    y_pred = df_train[prediction_column]
    y_true = df_train[true_label_column]
    
    # Compute Satisfying Metrics
    baccuracy_train = balanced_accuracy_score(y_true, y_pred)
    cohen_train = cohen_kappa_score(y_true, y_pred, weights="linear")

    # Validation
    y_pred = df_val[prediction_column]
    y_true = df_val[true_label_column]
    

    # Compute Satisfying Metrics
    baccuracy_val = balanced_accuracy_score(y_true, y_pred)
    cohen_val = cohen_kappa_score(y_true, y_pred, weights="linear")

    # Test
    y_pred = df_test[prediction_column]
    y_true = df_test[true_label_column]

    # Compute Satisfying Metrics
    baccuracy_test = balanced_accuracy_score(y_true, y_pred)
    cohen_test = cohen_kappa_score(y_true, y_pred, weights="linear")

    return baccuracy_train, cohen_train, baccuracy_val, cohen_val, baccuracy_test, cohen_test  

def make_predictions(input_file, model, variables, target_column, format_type = "csv"):
    """
    Make predictions using a sklearn model and put the true label in the same df

    Arguments:
    input_file (str) -- Path to .csv that has the features of the previously trained model
    model -- Sklearn model previously trained
    variables (str) -- Column Names of the variables
    format_type (str) -- Format type of the save data (csv or pickle)
    target_column -- Target column of the dataset 

    Returns:
    df -- Pandas Dataframe with thre columns  predictions, true label and split
    """
    # Read the data depending on source file
    if format_type == "csv":
        df = pd.read_csv(input_file)
    elif format_type == "pickle":
        df = pd.read_pickle(input_file)  

    # Extract Variables and target
    X = df[variables]
    y = df[target_column]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    # Make predictions
    predictions = model.predict(X)
    

    # Save predictions in a df
    y_pred_train = pd.DataFrame(predictions_train, columns = ["Prediction"])
    y_pred_test = pd.DataFrame(predictions_test, columns = ["Prediction"])

    # Create column that will refer to the split
    y_pred_train["split"] = "training"
    y_pred_test["split"] = "training"
    

    # concatenate dataframes
    df_train["true_label"] = y_train
    df_test = pd.concat([y_pred_test, y_test], axis=1)

    # Append df
    df = df_train.append(df_test) 

    return df

# Metrics Plots

def plot_roc_curve(model, X, y, title = "Receiver Operating Characteristic"):
    """
    Plot ROC curve

    Arguments:
    model -- Sklearn model already train
    X -- Data to predict on. The features has to be the same with which the model was trained on
    y -- target value of the X data
    title (str) -- title of the new plot

    Returns:
    matplotlib ROC curve
    """    

    probs = model.predict_proba(X)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y, preds) # calculate the fpr and tpr for all thresholds of the classification
    roc_auc = auc(fpr, tpr)

    # Graph
    plt.title(title)
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_confusion_matrix(input_file, split, true_label_column, prediction_column):
    """
    Plot Confusion Matrix

    Arguments:
    input_file (str) -- Path to .csv that has the predictions with columns and Prediction, True_Label, Split 
    split (str) -- training or validation
    Returns:
    Confusion Matrix matplotlib plot
    """      
    
    df = pd.read_csv(input_file)
    df = df[df['split'] == split]

    labels = ['diabetes', 'healthy']
    
    cm = confusion_matrix(df[true_label_column], df[prediction_column], normalize = 'true')
    cm = cm*100 # Multiply by 100 so te percentage is 99.7% instead of 0.997%
    cm = pd.DataFrame(cm, index=labels, columns=labels)        

    if split == "training":
        # Customize heatmap (Confusion Matrix)
        sns.set(font_scale = 1.5)
        ax = sns.heatmap(cm, cmap='BuGn', annot=True, annot_kws={"size": 18,"weight":"bold"}, cbar=False, fmt='.3g')
        for t in ax.texts: t.set_text(t.get_text() + " %") # Put percentage in confusion matrix
        plt.xlabel('Predicted',weight='bold',size=18)
        plt.ylabel('Known',weight='bold',size=18)
        plt.yticks(rotation=0,size=14) 
        plt.xticks(rotation=0 ,size=14)
    elif ((split == "validation") | (split == "testing") ):
        # Customize heatmap (Confusion Matrix)
        sns.set(font_scale = 1.5)
        ax = sns.heatmap(cm, cmap='BuGn', annot=True, annot_kws={"size": 18,"weight":"bold"}, cbar=False, fmt='.3g')
        for t in ax.texts: t.set_text(t.get_text() + " %") # Put percentage in confusion matrix
        plt.xlabel('Predicted',weight='bold',size=18)
        plt.yticks([])
        plt.xticks(rotation=0, size=14)


def plot_confusion_matrices(input_file, true_label_column, prediction_column):
    """
    Plot Confusion Matrices of training and validation set

    Arguments:
    input_file (str) -- Path to .csv that has the predictions with columns and Prediction, True_Label, Split 
    
    Returns:
    Confusion Matrix matplotlib plot
    """      
    
    plt.rcParams['figure.figsize'] = [18, 5]
    plt.subplot(1,3,1)
    plot_confusion_matrix(input_file, "training", true_label_column, prediction_column)
    plt.title("Matrix Training")
    plt.subplot(1,3,2)
    plot_confusion_matrix(input_file, "validation", true_label_column, prediction_column)
    plt.title("Matrix Validation")
    plt.subplot(1,3,3)
    plot_confusion_matrix(input_file, "testing", true_label_column, prediction_column)
    plt.title("Matrix Testing")


def plot_classification_report(input_file, split):
    """
    Plot Classification Report of sklearn

    Arguments:
    input_file (str) -- Path to .csv that has the predictions with columns and Prediction, True_Label, Split 
    split (str) -- training or validation
    Returns:
    Classification Report plot
    """        

    df = pd.read_csv(input_file)
    df = df[df['split'] == split]

    # Compute sklearn Classification Report and saves it in a pandas dataframe
    report = pd.DataFrame(classification_report(df.True_Label, df.Prediction, digits=3, output_dict=True)).transpose()
    report = report.loc[:, ["precision", "recall", "f1-score"]].drop(['accuracy', "macro avg"]) # Select what parts of the classification report to plot
    report = report*100 # Multiply by 100 so te percentage is 99.7% instead of 0.997%
    
    # Customize heatmap (Classification Report)
    sns.set(font_scale = 1.3)
    rdgn = sns.diverging_palette(h_neg=10, h_pos=130, s=80, l=62, sep=3, as_cmap=True)
    ax=sns.heatmap(report, cmap=rdgn, annot=True, annot_kws={"size": 14}, cbar=True, fmt='.3g', cbar_kws={'label':'%'}, center=90, vmin=0, vmax=100)
    ax.xaxis.tick_top()
    for t in ax.texts: t.set_text(t.get_text() + " %") #Put percentage 
    plt.yticks(rotation = 0)

def plot_classification_reports(input_file):
    """
    Plot Classification Reports of training and validation set

    Arguments:
    input_file (str) -- Path to .csv that has the predictions with columns and Prediction, True_Label, Split 
    Returns:
    Classification Report plot
    """        

    plt.rcParams['figure.figsize'] = [18, 5]
    plt.subplot(1,2,1)
    plot_classification_report(input_file, split="training")
    plt.title("Training")
    plt.subplot(1,2,2)
    plot_classification_report(input_file, split="validation")
    plt.title("Validation")

def print_accuracies(input_file):
    """
    Print accuracy and balanced accuracy

    Arguments:
    input_file (str) -- Path to .csv that has the predictions with columns and Prediction, True_Label, Split 
    Returns:
    accuracy and balanced accuracy
    """        

    df = pd.read_csv(input_file)
    splits = ["training", "validation"]

    for split in splits:
        df_intermediate = df[df['split'] == split]
        print("The " + str(split) + " accuracy is: ", round(accuracy_score(df_intermediate.True_Label, df_intermediate.Prediction)*100,2), "%")
        print("The " + str(split) + " balanced accuracy is: ", round(balanced_accuracy_score(df_intermediate.True_Label, df_intermediate.Prediction)*100,2), "%")
        print()
