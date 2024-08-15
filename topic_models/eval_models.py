###Test models using lt 

import linktransformer as lt
import pandas as pd
import numpy as np
import evaluate


def eval_predictions(df,label_col,pred_col,num_labels,averaging_type=None):
    print("***Test results***")
    metric0 = evaluate.load("accuracy")
    metric2 = evaluate.load("precision")
    metric1 = evaluate.load("recall")
    metric3 = evaluate.load("f1")

    preds=df[pred_col].values
    labels=df[label_col].values
    num_labels=df[label_col].nunique()


    results_dict={}
    results_dict["test/accuracy"]=metric0.compute(predictions=preds, references=labels)["accuracy"]

    if num_labels==2:
        averaging_type=None
        results_dict["test/precision"]=metric2.compute(predictions=preds, references=labels)["precision"]
        results_dict["test/recall"]=metric1.compute(predictions=preds, references=labels)["recall"]
        results_dict["test/f1"]=metric3.compute(predictions=preds, references=labels)["f1"]

    else:
        averaging_type=averaging_type
        results_dict["test/precision"]=metric2.compute(predictions=preds, references=labels,average=averaging_type)["precision"]
        results_dict["test/recall"]=metric1.compute(predictions=preds, references=labels,average=averaging_type)["recall"]
        results_dict["test/f1"]=metric3.compute(predictions=preds, references=labels,average=averaging_type)["f1"]
    
    print(results_dict)
    return results_dict

##load test data
test_data = pd.read_csv('/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/extra_data/all_topics/ww1/fine_tuning_data/test.csv', sep=',', header=0, index_col=0)

##Use lt to calculate accuracy

model_path="/mnt/122a7683-fa4b-45dd-9f13-b18cc4f4a187/deeprecordlinkage/linktransformer/extra_data/all_topics/ww1/models/retrained_w_large/checkpoint-720"
predictions_df=lt.classify_rows(test_data,on="article",model=model_path)


print(predictions_df)

##check accuracy
print(eval_predictions(predictions_df,label_col="label",pred_col="clf_preds_article",num_labels=2,averaging_type=None))
